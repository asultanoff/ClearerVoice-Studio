import yamlargparse, os, random, shutil, time
import numpy as np
import torch

from dataloader.dataloader import dataloader_wrapper
from dataloader.dataset_speech_onthefly import infer_speech_onthefly_target_speakers
from solver import Solver


def _prepare_checkpoint_dir(args):
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        return

    if args.evaluate_only or args.train_from_last_checkpoint:
        raise ValueError('checkpoint_dir is required for evaluation or checkpoint resume')

    args.checkpoint_dir = f"checkpoints/log_{time.strftime('%Y-%m-%d(%H:%M:%S)')}"
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    config_path = getattr(args, 'config', None)
    if isinstance(config_path, (list, tuple)):
        config_candidates = [path for path in config_path if isinstance(path, (str, os.PathLike))]
        config_path = config_candidates[-1] if config_candidates else None

    if config_path is not None:
        config_path = os.fspath(config_path)

    if config_path and os.path.isfile(config_path):
        shutil.copy(config_path, os.path.join(args.checkpoint_dir, 'config.yaml'))


def _has_partition_root(args, partition):
    partition_root = getattr(args, f'speech_{partition}_direc', None)
    return (partition_root is not None) or (getattr(args, 'audio_direc', None) is not None)


def _is_primary_process(args):
    return (args.distributed and args.local_rank == 0) or not args.distributed


def _load_project_env():
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return

    env_candidates = [
        find_dotenv(usecwd=True),
        os.path.join(os.path.dirname(__file__), '.env'),
    ]
    loaded = set()
    for env_path in env_candidates:
        if not env_path or env_path in loaded or not os.path.isfile(env_path):
            continue
        load_dotenv(env_path, override=False)
        loaded.add(env_path)


def _apply_hf_env_defaults(args):
    _load_project_env()
    if not getattr(args, 'hf_repo_id', None):
        repo_id = os.getenv('HF_REPO')
        if repo_id:
            args.hf_repo_id = repo_id.strip()


def _push_checkpoint_to_hub(args):
    repo_id = getattr(args, 'hf_repo_id', None)
    if not repo_id:
        return

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            'huggingface_hub is required to upload checkpoints. Run uv sync before enabling hub upload.'
        ) from exc

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f'Checkpoint directory does not exist: {checkpoint_dir}')

    api = HfApi()
    print(f'Ensuring Hugging Face Hub repo exists: {repo_id} (private={bool(args.hf_private)})')
    api.create_repo(repo_id=repo_id, repo_type='model', private=bool(args.hf_private), exist_ok=True)
    print(f'Uploading checkpoint artifacts from {checkpoint_dir} to https://huggingface.co/{repo_id}')
    api.upload_folder(
        repo_id=repo_id,
        repo_type='model',
        folder_path=checkpoint_dir,
        path_in_repo='',
        commit_message=f'Upload training artifacts from {os.path.basename(checkpoint_dir)}',
        ignore_patterns=['tensorboard/*', '**/__pycache__/*', '*.tmp'],
    )


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTORCH_SEED'] = str(args.seed)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    _apply_hf_env_defaults(args)
    _prepare_checkpoint_dir(args)
    device = torch.device('cuda') if args.use_cuda else torch.device('cpu')
    args.device = device
    is_primary = _is_primary_process(args)

    speech_loader = getattr(args, 'speech_loader', None) or 'manifest'
    if args.network_reference.cue == 'speech' and speech_loader == 'onthefly':
        inferred_speakers = infer_speech_onthefly_target_speakers(args)
        configured_speakers = getattr(args.network_audio, 'speakers', None)
        if configured_speakers in [None, 0]:
            args.network_audio.speakers = inferred_speakers
        elif configured_speakers != inferred_speakers:
            raise ValueError(
                f'network_audio.speakers={configured_speakers} does not match the '
                f'{inferred_speakers} target speakers discovered by the on-the-fly speech loader. '
                'Set network_audio.speakers to the inferred value or leave it unset.')

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', rank=args.local_rank, init_method='env://', world_size=args.world_size)

    from networks import network_wrapper
    model = network_wrapper(args)
    if args.distributed: model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)

    if is_primary:
        print("started on " + args.checkpoint_dir + '\n')
        print(args)
        print(f'Precision mode: {args.precision}')
        print(model)
        print("\nTotal number of parameters: {} \n".format(sum(p.numel() for p in model.parameters())))
        print("\nTotal number of trainable parameters: {} \n".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_learning_rate)

    train_sampler, train_generator = dataloader_wrapper(args,'train')
    val_generator = None
    if _has_partition_root(args, 'val'):
        _, val_generator = dataloader_wrapper(args, 'val')
    elif is_primary:
        print('No validation split configured; using the test split for epoch-level evaluation.')
    _, test_generator = dataloader_wrapper(args, 'test')
    args.train_sampler=train_sampler


    solver = Solver(args=args,
                model = model,
                optimizer = optimizer,
                train_data = train_generator,
                validation_data = val_generator,
                test_data = test_generator
                ) 
    if not args.evaluate_only:
        solver.train()

    # run evaluation script
    if is_primary:
        print("Start evaluation")
        args.batch_size=1
        args.max_length = 100
        args.distributed = False
        args.world_size = 1
        _, test_generator = dataloader_wrapper(args, 'test')
        solver.evaluate(test_generator)

        if not args.evaluate_only and getattr(args, 'hf_repo_id', None):
            print(f'Pushing checkpoints to Hugging Face Hub repo {args.hf_repo_id}')
            _push_checkpoint_to_hub(args)
            print('Finished pushing checkpoints to Hugging Face Hub')


if __name__ == '__main__':
    parser = yamlargparse.ArgumentParser("Settings")
    
    # Log and Visulization
    parser.add_argument('--seed', type=int)  
    parser.add_argument('--use_cuda', default=1, type=int, help='use cuda')

    parser.add_argument('--config', help='config file path', action=yamlargparse.ActionConfigFile) 
    parser.add_argument('--checkpoint_dir', type=str, help='the name of the log')
    parser.add_argument('--train_from_last_checkpoint', type=int, help='whether to train from a checkpoint, includes model weight, optimizer settings')
    parser.add_argument('--evaluate_only',  type=int, default=0, help='Only perform evaluation')

    # optimizer
    parser.add_argument('--loss_type', type=str, help='snr or sisnr')
    parser.add_argument('--init_learning_rate',  type=float, help='Init learning rate')
    parser.add_argument('--lr_warmup',  type=int, default=0, help='whether to perform lr warmup')
    parser.add_argument('--max_epoch', type=int, help='Number of maximum epochs')
    parser.add_argument('--clip_grad_norm',  type=float, help='Gradient norm threshold to clip')
    parser.add_argument('--precision', type=str, default='fp32', help='training precision: fp32 or bf16')

    # dataset settings
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--accu_grad',type=int, help='whether to accumulate grad')
    parser.add_argument('--effec_batch_size',type=int, help='effective Batch size')
    parser.add_argument('--max_length', type=int, help='max_length of mixture in training')
    parser.add_argument('--num_workers', type=int, help='Number of workers to generate minibatch')
    
    # network settings
    parser.add_argument('--causal', type=int, help='whether the newtwork is causal')
    parser.add_argument('--network_reference', type=dict, help='the nature of auxilary reference signal')
    parser.add_argument('--network_audio', type=dict, help='a dictionary that contains the network parameters')
    parser.add_argument('--init_from', type=str, help='whether to initilize the model weights from a pre-trained checkpoint')

    # others
    parser.add_argument('--mix_lst_path', type=str)
    parser.add_argument('--audio_direc', type=str)
    parser.add_argument('--reference_direc', type=str)
    parser.add_argument('--speaker_no', type=int)
    parser.add_argument('--audio_sr',  type=int, help='audio sampling_rate')
    parser.add_argument('--ref_sr',  type=int, help='reference signal sampling_rate')
    parser.add_argument('--speech_loader', type=str, help='manifest or onthefly')
    parser.add_argument('--speech_train_direc', type=str)
    parser.add_argument('--speech_val_direc', type=str)
    parser.add_argument('--speech_test_direc', type=str)
    parser.add_argument('--speech_glob_pattern', type=str)
    parser.add_argument('--speech_samples_per_epoch', type=int)
    parser.add_argument('--speech_eval_samples', type=int, help='fixed number of val/test samples to generate')
    parser.add_argument('--speech_eval_seed', type=int, help='seed used to freeze val/test on-the-fly mixtures')
    parser.add_argument('--speech_min_snr_db', type=float)
    parser.add_argument('--speech_max_snr_db', type=float)
    parser.add_argument('--speech_ref_max_length', type=float, help='maximum reference duration in seconds')
    parser.add_argument('--hf_repo_id', type=str, help='Hugging Face Hub model repo id; defaults to HF_REPO from .env')
    parser.add_argument('--hf_private', type=int, default=1, help='create the Hugging Face repo as private when missing')

    # Distributed training
    parser.add_argument("--local-rank", default=0, type=int)

    args = parser.parse_args()


    # check for single- or multi-GPU training
    args.distributed = False
    args.world_size = 1
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.world_size = int(os.environ['WORLD_SIZE'])
    assert torch.backends.cudnn.enabled, "cudnn needs to be enabled"

    main(args)
