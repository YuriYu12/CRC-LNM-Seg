import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from einops_exts import check_shape

from ddpm.text import tokenize, bert_embed
from ddpm.utils import *


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)


def make_ddim_timesteps(num_timesteps, ddim_timesteps, method="uniform"):
    if method == "uniform":
        c = num_timesteps // ddim_timesteps
        ddimsteps = torch.arange(0, num_timesteps, step=c)
    elif method == "quad":
        ddimsteps = (
            torch.linspace(0, torch.sqrt(torch.tensor(num_timesteps * 0.999)), steps=ddim_timesteps) ** 2
        ).long()
    return ddimsteps + 1


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        cfg,
        image_size,
        num_frames,
        channels=1,
        timesteps=1000,
        loss_type="l1",
        use_ddim=False,
        ddim_timesteps=250,
        ddim_eta=0.0,
        text_use_bert_cls=False,
    ):
        super().__init__()
        self.cfg = cfg
        self.denoise_fn = denoise_fn
        self.image_size = image_size
        self.num_frames = num_frames
        self.channels = channels

        self.loss_type = loss_type
        self.num_timesteps = timesteps

        self.text_use_bert_cls = text_use_bert_cls

        betas = cosine_beta_schedule(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # register buffer helper function that casts float64 to float32
        def register_buffer(name, val):
            return self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer("posterior_variance", posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer(
            "posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        # ddim
        self.use_ddim = use_ddim
        self.ddim_timesteps = ddim_timesteps
        self.ddim_eta = ddim_eta

        def cast(val):
            return val.to(torch.float32).to("cuda")

        if use_ddim:
            self.ddimsteps = make_ddim_timesteps(timesteps, ddim_timesteps)
            print(self.ddimsteps)

            self.ddim_alphas_bar = cast(self.alphas_cumprod[self.ddimsteps])
            self.ddim_alphas_bar_prev = cast(
                torch.cat([self.alphas_cumprod[:1], self.alphas_cumprod[self.ddimsteps[:-1]]])
            )
            self.ddim_sigmas = cast(
                ddim_eta
                * torch.sqrt((1 - self.ddim_alphas_bar_prev) / (1 - self.ddim_alphas_bar))
                * torch.sqrt(1 - self.ddim_alphas_bar / self.ddim_alphas_bar_prev)
            )
            print(self.ddim_alphas_bar)
            print(self.ddim_alphas_bar_prev)
            print(self.ddim_sigmas)

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_mean_variance(
        self,
        x,
        t,
        clip_denoised: bool,
        mask_cond=None,
        label_cond=None,
        multimask_cond=None,
        cond=None,
        cond_scale=1.0,
    ):
        if self.cfg.model.dist:
            noise = self.denoise_fn.module.forward_with_cond_scale(
                x,
                t,
                mask_cond=mask_cond,
                label_cond=label_cond,
                multimask_cond=multimask_cond,
                cond=cond,
                cond_scale=cond_scale,
            )
        else:
            noise = self.denoise_fn.forward_with_cond_scale(
                x,
                t,
                mask_cond=mask_cond,
                label_cond=label_cond,
                multimask_cond=multimask_cond,
                cond=cond,
                cond_scale=cond_scale,
            )

        x_recon = self.predict_start_from_noise(x, t=t, noise=noise)

        if clip_denoised:
            s = 1.0
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)

        return model_mean, posterior_variance, posterior_log_variance, noise, x_recon

    @torch.inference_mode()
    def p_sample(
        self,
        x,
        t,
        mask_cond=None,
        label_cond=None,
        cond=None,
        multimask_cond=None,
        cond_scale=1.0,
        clip_denoised=True,
    ):
        b, *_, device = *x.shape, x.device

        model_mean, _, model_log_variance, *_ = self.p_mean_variance(
            x=x,
            t=t,
            clip_denoised=clip_denoised,
            mask_cond=mask_cond,
            label_cond=label_cond,
            multimask_cond=multimask_cond,
            cond=cond,
            cond_scale=cond_scale,
        )

        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        result = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        return result

    @torch.inference_mode()
    def ddim_sample(
        self,
        x,
        t,
        idx,
        mask_cond=None,
        label_cond=None,
        cond=None,
        multimask_cond=None,
        cond_scale=1.0,
        clip_denoised=True,
    ):
        b, *_, device = *x.shape, x.device

        *_, epsilon, x_recon = self.p_mean_variance(
            x=x,
            t=t,
            clip_denoised=clip_denoised,
            mask_cond=mask_cond,
            label_cond=label_cond,
            multimask_cond=multimask_cond,
            cond=cond,
            cond_scale=cond_scale,
        )

        alpha_bar = extract(self.ddim_alphas_bar, idx, x.shape)
        alpha_bar_prev = extract(self.ddim_alphas_bar_prev, idx, x.shape)
        sigma = extract(self.ddim_sigmas, idx, x.shape)

        model_mean = x_recon * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev - sigma**2) * epsilon

        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        result = model_mean + nonzero_mask * sigma * noise
        return result

    @torch.inference_mode()
    def p_sample_loop(
        self,
        shape,
        mask_cond=None,
        label_cond=None,
        multimask_cond=None,
        cond=None,
        cond_scale=1.0,
    ):
        device = self.betas.device
        b = shape[0]

        img = torch.randn(shape, device=device)

        if self.use_ddim:
            for i, step in enumerate(tqdm(reversed(self.ddimsteps.tolist()), total=self.ddim_timesteps)):
                idx = self.ddim_timesteps - i - 1
                img = self.ddim_sample(
                    img,
                    torch.full((b,), step, device=device, dtype=torch.long),
                    torch.full((b,), idx, device=device, dtype=torch.long),
                    mask_cond=mask_cond,
                    label_cond=label_cond,
                    multimask_cond=multimask_cond,
                    cond=cond,
                    cond_scale=cond_scale,
                )
        else:
            for i in tqdm(reversed(range(0, self.num_timesteps)), total=self.num_timesteps):
                img = self.p_sample(
                    img,
                    torch.full((b,), i, device=device, dtype=torch.long),
                    mask_cond=mask_cond,
                    label_cond=label_cond,
                    multimask_cond=multimask_cond,
                    cond=cond,
                    cond_scale=cond_scale,
                )
        return img

    @torch.inference_mode()
    def sample(
        self,
        batch_size=1,
        mask_cond=None,
        label_cond=None,
        multimask_cond=None,
        cond=None,
        cond_scale=1.0,
    ):
        device = next(self.denoise_fn.parameters()).device
        shape = (batch_size, self.channels, self.num_frames, self.image_size, self.image_size)

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond)).to(device)

        samples = self.p_sample_loop(
            shape,
            mask_cond=mask_cond,
            label_cond=label_cond,
            multimask_cond=multimask_cond,
            cond=cond,
            cond_scale=cond_scale,
        )

        return samples

    def p_losses(
        self,
        x_start,
        t,
        mask_cond=None,
        label_cond=None,
        multimask_cond=None,
        cond=None,
        noise=None,
        **kwargs,
    ):
        b, c, f, h, w, device = *x_start.shape, x_start.device
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond), return_cls_repr=self.text_use_bert_cls)
            cond = cond.to(device)

        x_recon = self.denoise_fn(
            x_noisy,
            t,
            mask_cond=mask_cond,
            label_cond=label_cond,
            multimask_cond=multimask_cond,
            cond=cond,
            **kwargs,
        )

        if self.loss_type == "l1":  # MAE
            loss = F.l1_loss(noise, x_recon)
        elif self.loss_type == "l2":  # MSE
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, *args, **kwargs):
        b, device, img_size = x.shape[0], x.device, self.image_size
        check_shape(x, "b c f h w", c=self.channels, f=self.num_frames, h=img_size, w=img_size)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, *args, **kwargs)
