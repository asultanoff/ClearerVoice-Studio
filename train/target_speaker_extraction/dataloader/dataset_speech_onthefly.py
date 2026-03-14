import math
import random
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.utils.data as data

from .utils import DistributedSampler


def get_dataloader_speech_onthefly(args, partition):
    datasets = dataset_speech_onthefly(args, partition)
    shuffle = partition == 'train'

    sampler = DistributedSampler(
        datasets,
        num_replicas=args.world_size,
        rank=args.local_rank,
        shuffle=shuffle,
        seed=args.seed) if args.distributed else None

    generator = data.DataLoader(
        datasets,
        batch_size=1,
        shuffle=(sampler is None and shuffle),
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers != 0),
        sampler=sampler,
        collate_fn=custom_collate_fn)

    return sampler, generator


def custom_collate_fn(batch):
    a_mix, a_tgt, (aux, aux_len, speakers) = batch[0]
    a_mix = torch.tensor(a_mix)
    a_tgt = torch.tensor(a_tgt)
    aux = torch.tensor(aux)
    aux_len = torch.tensor(aux_len)
    speakers = torch.tensor(speakers)
    return a_mix, a_tgt, (aux, aux_len, speakers)


def infer_speech_onthefly_target_speakers(args):
    train_root = _resolve_partition_root(args, 'train')
    by_speaker = _scan_speech_root(train_root, _get_speech_glob_pattern(args))
    target_speakers = [speaker for speaker, paths in by_speaker.items() if len(paths) >= 2]
    if not target_speakers:
        raise ValueError(
            f'No target speakers with at least two utterances were found in {train_root}. '
            'The on-the-fly speech loader needs a separate target and reference utterance.')
    return len(target_speakers)


def _get_speech_glob_pattern(args):
    pattern = getattr(args, 'speech_glob_pattern', None)
    return pattern or '*.flac'


def _resolve_partition_root(args, partition):
    partition_root = getattr(args, f'speech_{partition}_direc', None)
    root_value = partition_root or getattr(args, 'audio_direc', None)
    if root_value is None:
        raise ValueError(
            f'No speech data directory configured for partition "{partition}". '
            f'Set speech_{partition}_direc or provide audio_direc as a fallback.')

    root = Path(root_value).expanduser()
    if not root.exists():
        raise FileNotFoundError(f'Speech data directory for partition "{partition}" does not exist: {root}')
    return root


def _scan_speech_root(root, pattern):
    by_speaker = defaultdict(list)

    for path in sorted(root.rglob(pattern)):
        if not path.is_file():
            continue

        stem = path.stem
        if '__' not in stem:
            continue

        speaker = stem.split('__', 1)[0].strip()
        if not speaker:
            continue

        by_speaker[speaker].append(path)

    if not by_speaker:
        raise ValueError(
            f'No files matching "{pattern}" with the "<speaker>__<index>" naming scheme were found in {root}.')

    return dict(by_speaker)


class dataset_speech_onthefly(data.Dataset):
    def __init__(self, args, partition):
        self.args = args
        self.partition = partition
        self.is_train = partition == 'train'

        self.audio_sr = args.audio_sr
        self.ref_sr = args.ref_sr
        self.batch_size = args.batch_size
        self.speaker_no = args.speaker_no
        self.max_audio_length = int(args.max_length * args.audio_sr)

        self.min_snr_db = float(getattr(args, 'speech_min_snr_db', -5.0))
        self.max_snr_db = float(getattr(args, 'speech_max_snr_db', 5.0))
        if self.min_snr_db > self.max_snr_db:
            raise ValueError('speech_min_snr_db must be smaller than or equal to speech_max_snr_db')

        ref_max_length = getattr(args, 'speech_ref_max_length', 0.0) or 0.0
        self.max_ref_length = int(ref_max_length * self.ref_sr) if ref_max_length > 0 else None

        if self.speaker_no < 2:
            raise ValueError('speaker_no must be at least 2 for the on-the-fly speech loader')

        self.root = _resolve_partition_root(args, partition)
        self.by_speaker = _scan_speech_root(self.root, _get_speech_glob_pattern(args))
        self.target_by_speaker = {
            speaker: paths for speaker, paths in self.by_speaker.items() if len(paths) >= 2
        }
        self.target_speakers = sorted(self.target_by_speaker)
        self.all_speakers = sorted(self.by_speaker)

        if not self.target_speakers:
            raise ValueError(
                f'No speakers with at least two utterances were found in {self.root}. '
                'The on-the-fly speech loader needs one utterance for the target and one for the reference.')
        if len(self.all_speakers) < self.speaker_no:
            raise ValueError(
                f'Found {len(self.all_speakers)} speakers in {self.root}, but speaker_no={self.speaker_no} '
                'requires at least that many distinct speakers to build a mixture.')

        self.speaker_dict = self._build_speaker_dict()
        self.samples_per_epoch = self._resolve_samples_per_epoch()
        self.fixed_batch_recipes = self._build_fixed_batch_recipes() if not self.is_train else None

    def _build_speaker_dict(self):
        if self.partition == 'train':
            target_speakers = self.target_speakers
        else:
            train_root = _resolve_partition_root(self.args, 'train')
            train_by_speaker = _scan_speech_root(train_root, _get_speech_glob_pattern(self.args))
            target_speakers = sorted(
                speaker for speaker, paths in train_by_speaker.items() if len(paths) >= 2
            )
        return {speaker: idx for idx, speaker in enumerate(target_speakers)}

    def _resolve_samples_per_epoch(self):
        if self.is_train:
            configured = getattr(self.args, 'speech_samples_per_epoch', None)
        else:
            configured = getattr(self.args, 'speech_eval_samples', None)
            if configured is None:
                configured = getattr(self.args, 'speech_samples_per_epoch', None)
        if configured is not None and configured > 0:
            return max(1, math.ceil(configured / self.batch_size))

        total_target_utterances = sum(len(paths) for paths in self.target_by_speaker.values())
        return max(1, math.ceil(total_target_utterances / self.batch_size))

    def _resolve_eval_seed(self):
        base_seed = getattr(self.args, 'speech_eval_seed', None)
        if base_seed is None:
            base_seed = getattr(self.args, 'seed', 0)
        partition_offset = {'val': 101, 'test': 202}.get(self.partition, 0)
        return int(base_seed) + partition_offset

    def _audioread(self, path, sampling_rate):
        data, fs = sf.read(path, dtype='float32')
        if data.ndim > 1:
            data = data[:, 0]
        if fs != sampling_rate:
            data = librosa.resample(data, orig_sr=fs, target_sr=sampling_rate)
        if data.size == 0:
            raise ValueError(f'Empty audio file encountered: {path}')
        return data.astype(np.float32)

    def _sample_segment(self, audio, length, segment_seed=None):
        if length is None:
            return audio
        if audio.shape[0] >= length:
            if segment_seed is None:
                start = random.randint(0, audio.shape[0] - length)
            else:
                start = random.Random(segment_seed).randint(0, audio.shape[0] - length)
            return audio[start:start + length]
        return np.pad(audio, (0, int(length - audio.shape[0])), mode='constant')

    def _scale_interferer(self, target, interferer, snr_db):
        target_power = np.mean(target ** 2) + 1e-8
        interferer_power = np.mean(interferer ** 2) + 1e-8
        scale = np.sqrt(target_power / (interferer_power * (10 ** (snr_db / 10.0))))
        return interferer * scale

    def _build_sample_recipe(self, rng):
        target_speaker = rng.choice(self.target_speakers)
        interferer_candidates = [speaker for speaker in self.all_speakers if speaker != target_speaker]
        interferer_speakers = rng.sample(interferer_candidates, self.speaker_no - 1)
        target_path, reference_path = rng.sample(self.target_by_speaker[target_speaker], 2)

        recipe = {
            'target_speaker': target_speaker,
            'target_path': target_path,
            'target_segment_seed': rng.randrange(1 << 63),
            'reference_path': reference_path,
            'reference_segment_seed': rng.randrange(1 << 63),
            'interferers': [],
        }
        for interferer_speaker in interferer_speakers:
            recipe['interferers'].append({
                'path': rng.choice(self.by_speaker[interferer_speaker]),
                'segment_seed': rng.randrange(1 << 63),
                'snr_db': rng.uniform(self.min_snr_db, self.max_snr_db),
            })
        return recipe

    def _build_fixed_batch_recipes(self):
        rng = random.Random(self._resolve_eval_seed())
        fixed_batches = []
        for _ in range(self.samples_per_epoch):
            fixed_batches.append([self._build_sample_recipe(rng) for _ in range(self.batch_size)])
        return fixed_batches

    def _materialize_sample(self, recipe):
        target_audio = self._sample_segment(
            self._audioread(recipe['target_path'], self.audio_sr),
            self.max_audio_length,
            segment_seed=recipe['target_segment_seed'])
        reference_audio = self._sample_segment(
            self._audioread(recipe['reference_path'], self.ref_sr),
            self.max_ref_length,
            segment_seed=recipe['reference_segment_seed'])

        mixture_audio = target_audio.copy()
        for interferer in recipe['interferers']:
            interferer_audio = self._sample_segment(
                self._audioread(interferer['path'], self.audio_sr),
                self.max_audio_length,
                segment_seed=interferer['segment_seed'])
            mixture_audio += self._scale_interferer(target_audio, interferer_audio, interferer['snr_db'])

        max_abs = np.max(np.abs(mixture_audio))
        if max_abs > 1:
            mixture_audio = mixture_audio / max_abs
            target_audio = target_audio / max_abs

        return (
            mixture_audio.astype(np.float32),
            target_audio.astype(np.float32),
            reference_audio.astype(np.float32),
            recipe['target_speaker'],
        )

    def __getitem__(self, index):
        mix_audios = []
        tgt_audios = []
        tgt_refs = []
        speakers = []
        max_ref_length = 0

        if self.is_train:
            batch_recipes = [self._build_sample_recipe(random) for _ in range(self.batch_size)]
        else:
            batch_recipes = self.fixed_batch_recipes[index]

        for recipe in batch_recipes:
            mixture_audio, target_audio, reference_audio, target_speaker = self._materialize_sample(recipe)

            mix_audios.append(mixture_audio)
            tgt_audios.append(target_audio)
            tgt_refs.append(reference_audio)
            max_ref_length = max(max_ref_length, reference_audio.shape[0])

            if self.partition == 'test':
                speakers.append(-1)
            else:
                speakers.append(self.speaker_dict.get(target_speaker, -1))

        aux_length = []
        aux = []
        for ref in tgt_refs:
            length = ref.shape[0]
            aux_length.append(length)
            ref = np.pad(ref, (0, int(max_ref_length - length)), mode='edge')
            aux.append(ref)

        return (
            np.asarray(mix_audios, dtype=np.float32),
            np.asarray(tgt_audios, dtype=np.float32),
            (np.asarray(aux, dtype=np.float32), np.asarray(aux_length), speakers),
        )

    def __len__(self):
        return self.samples_per_epoch
