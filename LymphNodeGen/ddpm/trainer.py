import copy
import numpy as np
from pathlib import Path
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler

from ddpm.utils import *


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)


class Trainer(object):
    def __init__(
        self,
        diffusion,
        cfg,
        dataset=None,
        valdataset=None,
        ema_decay=0.995,
        step_start_ema=2000,
        update_ema_every=10,
        train_lr=1e-4,
        train_batch_size=32,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        save_and_sample_every=1000,
        amp=False,
        results_folder="",
        num_samples=2,
        max_grad_norm=None,
        num_workers=20,
        logger=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.model = diffusion
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema

        self.step = 0
        self.batch_size = train_batch_size
        self.train_num_steps = train_num_steps
        self.gradient_accumulate_every = gradient_accumulate_every
        self.save_and_sample_every = save_and_sample_every

        self.ds = dataset
        self.vds = valdataset if valdataset else self.ds
        print(f"{len(self.ds)} training cases, {len(self.vds)} validation cases.")

        if cfg.model.dist:
            dl = DataLoader(
                self.ds, batch_size=train_batch_size, num_workers=num_workers, sampler=DistributedSampler(self.ds)
            )
        else:
            dl = DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)

        self.dl = cycle(dl)

        self.opt = Adam(self.model.parameters(), lr=train_lr)

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self.max_grad_norm = max_grad_norm

        self.num_samples = num_samples
        self.logger = logger
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
            "scaler": self.scaler.state_dict(),
            "opt": self.opt.state_dict(),
        }
        torch.save(data, str(self.results_folder / f"model-{milestone}.pt"))

    def load(self, milestone, pretrained=False, map_location=None):
        if not milestone.startswith("/nas"):
            milestone = str(self.results_folder / f"model-{milestone}.pt")

        print(f"resume from: {milestone}")
        data = torch.load(milestone, map_location=map_location)

        if pretrained:
            self.model.load_state_dict(data["model"])
            self.ema_model.load_state_dict(data["ema"])
        else:
            self.step = data["step"]
            self.model.load_state_dict(data["model"])
            self.ema_model.load_state_dict(data["ema"])
            self.scaler.load_state_dict(data["scaler"])
            try:
                self.opt.load_state_dict(data["opt"])
            except:
                print("no opt state dict")

    def train(self, prob_focus_present=0.0, focus_present_mask=None):
        while self.step < self.train_num_steps:
            loss_log = []
            if not self.cfg.model.use_ema:
                self.model.train()
            for i in range(self.gradient_accumulate_every):
                loader = next(self.dl)
                data = loader["data"].cuda()

                if self.cfg.model.use_multimask:
                    multimask = loader["multi"].cuda()
                    with autocast(enabled=self.amp):
                        loss = self.model(
                            data,
                            multimask_cond=multimask,
                            prob_focus_present=prob_focus_present,
                            focus_present_mask=focus_present_mask,
                            cond_drop_rate=self.cfg.model.cond_drop_rate,
                        )
                        self.scaler.scale(loss / self.gradient_accumulate_every).backward()
                else:
                    if self.cfg.model.use_mask:
                        mask = loader["mask"].cuda()
                        with autocast(enabled=self.amp):
                            loss = self.model(
                                data,
                                mask_cond=mask,
                                prob_focus_present=prob_focus_present,
                                focus_present_mask=focus_present_mask,
                                cond_drop_rate=self.cfg.model.cond_drop_rate,
                            )
                            self.scaler.scale(loss / self.gradient_accumulate_every).backward()
                    else:
                        with autocast(enabled=self.amp):
                            loss = self.model(
                                data, prob_focus_present=prob_focus_present, focus_present_mask=focus_present_mask
                            )
                            self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                print(f"{self.step}: {loss.item()}")
                loss_log.append(loss.item())

            self.logger.add_scalar("train_loss", np.mean(loss_log), self.step)
            log_function({"Step %d loss" % self.step: loss_log}, str(self.results_folder / f"log.txt"))

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.cfg.model.local_rank == 0 and (self.step + 1) % self.save_and_sample_every == 0:
                milestone = (self.step + 1) // self.save_and_sample_every

                if self.cfg.model.use_ema:
                    self.ema_model.eval()
                else:
                    self.model.eval()
                with torch.no_grad():
                    cond = self.vds[0]
                    x_input = cond["data"].unsqueeze(0).cuda()
                    cond_1 = self.vds[1]
                    x_input_1 = cond_1["data"].unsqueeze(0).cuda()

                    if self.cfg.model.use_multimask:
                        multimask_cond = cond["multi"].unsqueeze(0).cuda()
                        multimask_cond_1 = cond_1["multi"].unsqueeze(0).cuda()
                        all_videos_list = self.ema_model.sample(
                            batch_size=self.num_samples,
                            multimask_cond=torch.cat([multimask_cond, multimask_cond_1], dim=0),
                            image=torch.cat([x_input, x_input_1], dim=0),
                        )
                        all_videos_list = torch.cat(
                            [
                                all_videos_list,
                                torch.cat([cond["mask"].unsqueeze(0), cond_1["mask"].unsqueeze(0)], dim=0).cuda(),
                            ],
                            dim=0,
                        )
                    else:
                        if self.cfg.model.use_mask:
                            mask_cond = cond["mask"].unsqueeze(0).cuda()
                            mask_cond_1 = cond_1["mask"].unsqueeze(0).cuda()
                            all_videos_list = self.ema_model.sample(
                                batch_size=self.num_samples,
                                mask_cond=torch.cat([mask_cond, mask_cond_1], dim=0),
                            )
                            all_videos_list = torch.cat(
                                [all_videos_list, torch.cat([mask_cond, mask_cond_1], dim=0)], dim=0
                            )
                        else:
                            all_videos_list = self.ema_model.sample(batch_size=self.num_samples)
                            all_videos_list = torch.cat(all_videos_list, dim=0)

                all_videos_list = F.pad(all_videos_list, (2, 2, 2, 2))
                all_videos_list = rearrange(all_videos_list, "(i j) c f h w -> c f (i h) (j w)", i=self.num_samples)
                tensor_to_gif(all_videos_list, str(self.results_folder / f"{milestone}.gif"))

                self.save("latest")
                if milestone % 10 == 0:
                    self.save(milestone)

            self.step += 1
            torch.cuda.empty_cache()
