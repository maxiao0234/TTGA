import os
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
import torch.nn.functional as nnf
import numpy as np
from torch.optim.adam import Adam
from PIL import Image

import utils


def one_step_optimization(args):
    args.device = torch.device('cuda')
    ldm_stable = StableDiffusionPipeline.from_pretrained(args.model_name).to(args.device)
    if os.path.exists(args.lora_dir):
        ldm_stable.load_lora_weights(args.lora_dir)
    else:
        print('Invalid LoRA Configuration!')

    tokenizer = ldm_stable.tokenizer
    text_encoder = ldm_stable.text_encoder
    vae = ldm_stable.vae.eval()
    unet = ldm_stable.unet.eval()

    input_ids = tokenizer([args.prompt, ""], padding="max_length", return_tensors="pt").input_ids
    text_embedding, uncond_embedding = text_encoder(input_ids.to(args.device))[0].chunk(2)

    args.betas = torch.linspace(args.beta_start**0.5, args.beta_end**0.5, 1000, dtype=torch.float32) ** 2
    args.alphas = 1.0 - args.betas
    args.alphas_cumprod = torch.cumprod(args.alphas, dim=0)
    timesteps_forward = torch.arange(0, args.t_start + 1, args.t_interval_forward).to(args.device)

    dataset_list = os.listdir(args.dataset_root)
    for d, dataset in enumerate(dataset_list):
        dataset_str = ['*'] * len(dataset_list)
        dataset_str[d] = dataset
        dataset_str = str(dataset_str).replace('\'', '')

        image_dir = os.path.join(args.dataset_root, dataset, 'images')
        output_dir = os.path.join(args.output_root, dataset)
        result_dir = os.path.join(args.result_root, dataset)
        images = [os.path.join(image_dir, image_file) for image_file in os.listdir(image_dir)]
        images = sorted(images)

        for i in range(len(images)):
            image_path = images[i]
            image_name = os.path.basename(image_path)
            image_name = os.path.splitext(image_name)[0]
            save_dir = os.path.join(result_dir, image_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            image = Image.open(image_path).resize((args.resolution, args.resolution))
            original_samples = utils.image2latent(vae, image)

            with torch.no_grad():
                latent_start = original_samples.clone()
                for j in range(len(timesteps_forward) - 1): 
                    noise_pred_text = unet(latent_start, timesteps_forward[j], encoder_hidden_states=text_embedding)["sample"]
                    latent_start = utils.add_noise(args, latent_start, noise_pred_text, timesteps_forward[j], timesteps_forward[j + 1])
                noise_pred_text = unet(latent_start, timesteps_forward[-1], encoder_hidden_states=text_embedding)["sample"]
            
            null_text_embedding = uncond_embedding.clone().detach()
            null_text_embedding.requires_grad = True
            optimizer = Adam([null_text_embedding], lr=args.lr)

            progress_bar = tqdm(range(args.maximum_train_steps), total=args.maximum_train_steps)
            progress_bar.set_description(f'{dataset_str}: [{i + 1} | {len(images)}]')
            for j in progress_bar:
                noise_pred_null_text = unet(latent_start, timesteps_forward[-1], encoder_hidden_states=null_text_embedding)["sample"]
                noise_pred = noise_pred_null_text + args.guidance_scale * (noise_pred_text - noise_pred_null_text)
                pred_original_samples = utils.denoise(args, latent_start, noise_pred, timesteps_forward[-1], timesteps_forward[0], eta=0.0)

                loss = nnf.mse_loss(pred_original_samples, original_samples)
                progress_bar.set_postfix(loss=loss.item())
                if j > args.minimum_train_steps and loss.item() < args.early_stop_loss + timesteps_forward[-1] * 1e-6:
                    progress_bar.close()
                    break
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                
            torch.save(latent_start, os.path.join(output_dir, f'latent_{args.t_start}_{image_name}.pt'))
            torch.save(null_text_embedding, os.path.join(output_dir, f'null_text_{args.t_start}_{image_name}.pt'))


def augmentation(args):
    args.device = torch.device('cuda')
    ldm_stable = StableDiffusionPipeline.from_pretrained(args.model_name).to(args.device)
    if os.path.exists(args.lora_dir):
        ldm_stable.load_lora_weights(args.lora_dir)
    else:
        print('Invalid LoRA Configuration!')

    tokenizer = ldm_stable.tokenizer
    text_encoder = ldm_stable.text_encoder
    vae = ldm_stable.vae.eval()
    unet = ldm_stable.unet.eval()
    controller = utils.AttentionStore()
    utils.register_attention_control(unet, controller)

    input_ids = tokenizer([args.prompt, ""], padding="max_length", return_tensors="pt").input_ids
    text_embedding, uncond_embedding = text_encoder(input_ids.to(args.device))[0].chunk(2)

    args.betas = torch.linspace(args.beta_start**0.5, args.beta_end**0.5, 1000, dtype=torch.float32) ** 2
    args.alphas = 1.0 - args.betas
    args.alphas_cumprod = torch.cumprod(args.alphas, dim=0)
    timesteps_denoising = torch.arange(args.t_start, -1, -args.t_interval_denoising).to(args.device)

    dataset_list = os.listdir(args.dataset_root)
    for d, dataset in enumerate(dataset_list):
        dataset_str = ['*'] * len(dataset_list)
        dataset_str[d] = dataset
        dataset_str = str(dataset_str).replace('\'', '')

        image_dir = os.path.join(args.dataset_root, dataset, 'images')
        output_dir = os.path.join(args.output_root, dataset)
        result_dir = os.path.join(args.result_root, dataset)
        images = [os.path.join(image_dir, image_file) for image_file in os.listdir(image_dir)]
        images = sorted(images)

        progress_bar = tqdm(range(len(images)), total=len(images))
        progress_bar.set_description(f'{dataset_str}')
        for i in progress_bar:
            image_path = images[i]
            image_name = os.path.basename(image_path)
            image_name = os.path.splitext(image_name)[0]
            save_dir = os.path.join(result_dir, image_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            image = Image.open(image_path).resize((args.resolution, args.resolution))
            null_text_embedding = torch.load(os.path.join(output_dir, f'null_text_{args.t_start}_{image_name}.pt')).to(args.device)
            latent_start = torch.load(os.path.join(output_dir, f'latent_{args.t_start}_{image_name}.pt')).to(args.device)

            for n in range(args.num_aug_images):
                latent_cur = latent_start.clone()
                latent_cur = utils.aug_one_image(args, controller, unet, latent_cur, timesteps_denoising, uncond_embedding, text_embedding, null_text_embedding)
                gen_aug_image = utils.latent2image(vae, latent_cur)[0]
                # gen_aug_image = np.concatenate([np.array(image), gen_aug_image], axis=1)
                Image.fromarray(gen_aug_image).save(os.path.join(save_dir, f'{n + 1}.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TTGA training and evaluation script', add_help=False)
    parser.add_argument('--project-name', default='TTGA', type=str)
    parser.add_argument('--model-name', default='runwayml/stable-diffusion-v1-5', type=str)
    parser.add_argument('--lora-dir', default='', type=str)
    parser.add_argument('--prompt', default='', type=str)
    parser.add_argument('--dataset-root', default='', type=str)
    parser.add_argument('--output-root', default='', type=str)
    parser.add_argument('--result-root', default='', type=str)
    parser.add_argument('--resolution', default=512, type=int)
    parser.add_argument('--num-aug-images', default=10, type=int)

    # DDIM
    parser.add_argument('--beta-start', default=0.0001, type=float)
    parser.add_argument('--beta-end', default=0.02, type=float)
    parser.add_argument('--guidance-scale', default=2, type=float)
    parser.add_argument('--guidance-scale-c', default=1, type=float)
    parser.add_argument('--guidance-scale-r', default=1.2, type=float)
    parser.add_argument('--drop-p', default=0.0, type=float)

    # Optimizer
    parser.add_argument('--t-start', default=300, type=int)
    parser.add_argument('--t-interval-forward', default=10, type=int)
    parser.add_argument('--t-interval-denoising', default=100, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--maximum-train-steps', default=1000, type=int)
    parser.add_argument('--minimum-train-steps', default=100, type=int)
    parser.add_argument('--early_stop_loss', default=1e-4, type=float)

    args = parser.parse_args()

    one_step_optimization(args)
    augmentation(args)

