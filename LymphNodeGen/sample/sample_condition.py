import os
import sys
import hydra
import torch
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from pathlib import Path
from omegaconf import DictConfig, open_dict

sys.path.append(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.get_dataset import get_dataset
from ddpm import Trainer, GaussianDiffusion, UNetModelLDM


@hydra.main(config_path="../config", config_name="base_cfg", version_base=None)
def run(cfg: DictConfig):
    with open_dict(cfg):
        cfg.model.samples_folder = os.path.join(
            cfg.model.samples_folder, cfg.dataset.name, cfg.model.samples_folder_postfix
        )

    samples_folder_image = Path(os.path.join(cfg.model.samples_folder, "image"))
    samples_folder_image.mkdir(exist_ok=True, parents=True)
    samples_folder_mask = Path(os.path.join(cfg.model.samples_folder, "mask"))
    samples_folder_mask.mkdir(exist_ok=True, parents=True)

    train_dataset, val_dataset, *_ = get_dataset(cfg)

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
        ).cuda()

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

    trainer = Trainer(
        diffusion,
        cfg=cfg,
        dataset=train_dataset,
        valdataset=val_dataset,
        ema_decay=cfg.model.ema_decay,
        train_batch_size=cfg.model.batch_size,
        train_lr=cfg.model.train_lr,
        train_num_steps=cfg.model.train_num_steps,
        gradient_accumulate_every=cfg.model.gradient_accumulate_every,
        save_and_sample_every=cfg.model.save_and_sample_every,
        amp=cfg.model.amp,
        num_samples=cfg.model.num_samples,
        results_folder=cfg.model.results_folder,
        num_workers=cfg.model.num_workers,
    )

    trainer.load(cfg.model.ddpm_ckpt, map_location="cuda:0")

    for i in tqdm(range(cfg.model.samples), desc="sampling process"):
        print("idx:", i)
        mask = np.load(
            os.path.join(
                cfg.dataset.root_dir.replace("image", "mask"), "Cond_%s_%04d_mask.npy" % (cfg.dataset.name, i + 1)
            )
        )
        mask_cond = mask[np.newaxis, np.newaxis, :]
        mask_cond = torch.from_numpy(mask_cond).float().cuda()

        if cfg.model.use_multimask:
            image = np.load(os.path.join(cfg.dataset.root_dir, "Cond_%s_%04d_image.npy" % (cfg.dataset.name, i + 1)))
            image_cond = image[np.newaxis, np.newaxis, :]
            image_cond = torch.from_numpy(image_cond).float().cuda()
            multi = np.load(
                os.path.join(
                    cfg.dataset.root_dir.replace("image", "supp"), "Cond_%s_%04d_multi.npy" % (cfg.dataset.name, i + 1)
                )
            )
            multi_cond = multi[np.newaxis, :]
            multi_cond = torch.from_numpy(multi_cond).float().cuda()

            sample = trainer.ema_model.sample(
                batch_size=cfg.model.batch_size,
                multimask_cond=multi_cond,
                image=image_cond,
                cond_scale=cfg.model.cond_scale,
            )

        elif cfg.model.use_mask:
            sample = trainer.ema_model.sample(
                batch_size=cfg.model.batch_size,
                mask_cond=mask_cond,
                cond_scale=cfg.model.cond_scale,
            )

        if cfg.dataset.name == "LNM":
            sample = sample * 180.0 + 60.0  # [-1,1]->[-120,240]

        sample_image = sitk.GetImageFromArray(sample[0][0].cpu())
        sample_mask = sitk.GetImageFromArray(mask)
        sitk.WriteImage(
            sample_image, os.path.join(samples_folder_image, "Syn_%s_%04d_image.nii.gz" % (cfg.dataset.name, i + 1))
        )
        sitk.WriteImage(
            sample_mask, os.path.join(samples_folder_mask, "Syn_%s_%04d_mask.nii.gz" % (cfg.dataset.name, i + 1))
        )


if __name__ == "__main__":
    run()
