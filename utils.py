from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
import torch.nn.functional as F
import numpy as np
import abc


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).detach().numpy()
    image = (image * 255).astype(np.uint8)
    return image


def image2latent(vae, image):
    image = torch.from_numpy(np.array(image)).float() / 127.5 - 1
    image = image.permute(2, 0, 1).unsqueeze(0).to(vae.device, dtype=vae.dtype)
    latents = vae.encode(image)['latent_dist'].mean
    latents = latents * 0.18215
    return latents


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        # return self.num_att_layers if LOW_RESOURCE else 0
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if not self.store:
            return attn
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0]
            attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.store = False


class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if self.store:
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
                self.step_store[key].append(attn)
            return attn
        else:
            return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            context = encoder_hidden_states
            mask = attention_mask
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.head_to_batch_dim(q)
            k = self.head_to_batch_dim(k)
            v = self.head_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)

            return to_out(out)

        return forward

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count


def aggregate_attention(attention_store: AttentionStore, prompts, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]

    map = out[:, :, 1: ]
    map = map.max(dim=-1)[0]
    return map


def add_noise(args, sample, noise, timestep, next_step):
    alpha_prod_t = args.alphas_cumprod[timestep].to(args.device)
    alpha_prod_t_next = args.alphas_cumprod[next_step].to(args.device)
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * noise) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * noise
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def denoise(args, sample, noise_pred, timestep, prev_timestep, eta=0.0):
    alpha_prod_t = args.alphas_cumprod[timestep]
    alpha_prod_t_prev = args.alphas_cumprod[prev_timestep]
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    std_dev_t = eta * variance ** (0.5)

    pred_original_sample = (sample - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * noise_pred
    prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

    variance_noise = torch.randn(noise_pred.shape).to(args.device)
    prev_sample = prev_sample + std_dev_t * variance_noise

    return prev_sample


@torch.no_grad()
def aug_one_image(args, controller, unet, latent_start, timesteps, uncond_embedding, text_embedding, null_text_embedding):
    noise_pred_text = unet(latent_start, timesteps[0], encoder_hidden_states=text_embedding)["sample"]
    noise_pred_null_text = unet(latent_start, timesteps[0], encoder_hidden_states=null_text_embedding)["sample"]
    noise_pred_start = noise_pred_null_text + args.guidance_scale * (noise_pred_text - noise_pred_null_text)
    
    prev_store = controller.store
    controller.store = True

    s_uncond = 1 - args.guidance_scale_c
    s_text = args.guidance_scale_c - args.guidance_scale_r * (1 - args.guidance_scale)
    s_null = args.guidance_scale_r * (1 - args.guidance_scale)

    latent_cur = latent_start.clone()
    # adj_noise = torch.randn(1, 4, 1, 1).to(args.device) * 0.2
    # adj_noise = F.upsample(adj_noise, size=(args.resolution // 8, args.resolution // 8), mode='bilinear', align_corners=False)
    for i in range(len(timesteps) - 1):
        t_cur = timesteps[i]
        t_prev = timesteps[i + 1]

        controller.reset()
        noise_pred_text = unet(latent_cur, t_cur, encoder_hidden_states=text_embedding)["sample"]
        map_text = aggregate_attention(controller, [args.prompt], res=args.resolution // 32, from_where=["up", "down"], is_cross=True, select=0)
        map_text = torch.nn.functional.interpolate(map_text.unsqueeze(0).unsqueeze(0), size=(args.resolution // 8, args.resolution // 8), mode='bilinear')
        
        controller.reset()
        noise_pred_null_text = unet(latent_cur, t_cur, encoder_hidden_states=null_text_embedding)["sample"]
        map_null_text = aggregate_attention(controller, [args.prompt], res=args.resolution // 32, from_where=["up", "down"], is_cross=True, select=0)
        map_null_text = torch.nn.functional.interpolate(map_null_text.unsqueeze(0).unsqueeze(0), size=(args.resolution // 8, args.resolution // 8), mode='bilinear')

        noise_pred = s_text * noise_pred_text + s_null * noise_pred_null_text
        map_pred = s_text * map_text + s_null * map_null_text

        if args.guidance_scale_c != 1.:
            controller.reset()
            noise_pred_uncond = unet(latent_cur, t_cur, encoder_hidden_states=uncond_embedding)["sample"]
            map_uncond = aggregate_attention(controller, [args.prompt], res=args.resolution // 32, from_where=["up", "down"], is_cross=True, select=0)
            map_uncond = torch.nn.functional.interpolate(map_uncond.unsqueeze(0).unsqueeze(0), size=(args.resolution // 8, args.resolution // 8), mode='bilinear')

            noise_pred = noise_pred + s_uncond * noise_pred_uncond
            map_pred = map_pred + s_uncond * map_uncond

        drop_mask = torch.rand(1, 1, args.resolution // 8, args.resolution // 8).to(args.device)
        drop_mask = F.upsample(drop_mask, size=(args.resolution // 8, args.resolution // 8), mode='bilinear', align_corners=False).squeeze(1)
        drop_mask = torch.where(drop_mask >= args.drop_p, 1., 0.).to(args.device)
        mask_id = torch.where(map_pred < 0, 0, 1)
        mask_id = mask_id * drop_mask
        mask_aug = 1 - mask_id

        latent_id = denoise(args, latent_start, noise_pred_start, timesteps[0], t_prev, eta=0.0)
        latent_aug = denoise(args, latent_cur, noise_pred, t_cur, t_prev, eta=1.0)
        latent_cur = latent_id * mask_id + latent_aug * mask_aug

    controller.store = prev_store
    return latent_cur