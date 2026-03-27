import os
import sys
import hydra
import random
import warnings
import numpy as np
from omegaconf import DictConfig, open_dict
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter

sys.path.append(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.get_dataset import get_dataset
from ddpm import Trainer, GaussianDiffusion, UNetModelLDM

warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def model_size_MB(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_size_bytes = total_params * 4  # assuming float32 parameters
    total_size_MB = total_size_bytes / (1024**2)  # convert bytes to MB
    return total_size_MB


@hydra.main(config_path="../config", config_name="base_cfg", version_base=None)
def run(cfg: DictConfig):
    set_seed(1234)  # set seed

    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, "ddpm", cfg.model.results_folder_postfix
        )

    if cfg.model.dist:  # set dist
        dist.init_process_group(backend="nccl", init_method="env://")
        cfg.model.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(cfg.model.local_rank)
        print("local_rank:", cfg.model.local_rank)

    if cfg.model.denoising_net == "LDM":
        model = UNetModelLDM(
            cfg=cfg,
            dims=3,
            image_size=cfg.model.diffusion_img_size,
            depth_size=cfg.model.diffusion_depth_size,
            in_channels=cfg.model.diffusion_num_channels,
            out_channels=cfg.model.diffusion_num_channels,
            model_channels=cfg.model.dim,
            num_res_blocks=cfg.model.num_res_blocks,
            attention_resolutions=cfg.model.attention_resolutions,
            num_heads=cfg.model.num_heads,
            num_classes=cfg.model.num_classes,
            use_spatial_transformer=cfg.model.use_spatial_transformer,
            transformer_depth=cfg.model.transformer_depth,
            context_dim=cfg.model.context_dim,
            use_checkpoint=True,
            use_scale_shift_norm=True,
            resblock_updown=False,
        ).cuda()
    
    print(f"model size: {model_size_MB(model)} MB")

    if cfg.model.dist:
        model = DistributedDataParallel(model, device_ids=[cfg.model.local_rank])

    diffusion = GaussianDiffusion(
        model,
        cfg=cfg,
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        loss_type=cfg.model.loss_type,
        use_ddim=cfg.model.use_ddim,
        ddim_timesteps=cfg.model.ddim_timesteps,
        ddim_eta=cfg.model.ddim_eta,
    ).cuda()

    train_dataset, val_dataset, *_ = get_dataset(cfg)

    writer = SummaryWriter(log_dir=os.path.join(cfg.model.results_folder, "log"))

    trainer = Trainer(
        diffusion,
        cfg=cfg,
        dataset=train_dataset,
        valdataset=val_dataset,
        ema_decay=cfg.model.ema_decay,
        train_lr=cfg.model.train_lr,
        train_batch_size=cfg.model.batch_size,
        train_num_steps=cfg.model.train_num_steps,
        gradient_accumulate_every=cfg.model.gradient_accumulate_every,
        save_and_sample_every=cfg.model.save_and_sample_every,
        amp=cfg.model.amp,
        num_samples=cfg.model.num_samples,
        results_folder=cfg.model.results_folder,
        num_workers=cfg.model.num_workers,
        logger=writer,
    )

    if cfg.model.load_milestone:
        trainer.load(
            cfg.model.load_milestone,
            pretrained=cfg.model.finetune,
            map_location=torch.device(f"cuda:{cfg.model.local_rank}"),
        )

    trainer.train()

    writer.close()

    if cfg.model.dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    run()
