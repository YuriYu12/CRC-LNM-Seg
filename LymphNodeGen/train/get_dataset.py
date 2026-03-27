from dataset import LNMDataset


def get_dataset(cfg):
    if cfg.dataset.name == "LNM":
        train_dataset = LNMDataset(
            root_dir=cfg.dataset.root_dir,
            image_size=cfg.model.diffusion_img_size,
            depth_size=cfg.model.diffusion_depth_size,
            multi=cfg.model.use_multimask,
            flag="train",
        )
        val_dataset = LNMDataset(
            root_dir=cfg.dataset.root_dir,
            image_size=cfg.model.diffusion_img_size,
            depth_size=cfg.model.diffusion_depth_size,
            multi=cfg.model.use_multimask,
            flag="val",
        )
        sample_dataset = LNMDataset(
            root_dir=cfg.dataset.root_dir,
            image_size=cfg.model.diffusion_img_size,
            depth_size=cfg.model.diffusion_depth_size,
            multi=cfg.model.use_multimask,
            flag="sample",
        )
        sampler = None
        return train_dataset, val_dataset, sample_dataset, sampler
