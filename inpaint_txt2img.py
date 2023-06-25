import argparse
import torch
import PIL
from io import BytesIO
import requests
import wandb
from diffusers import RePaintScheduler
from modules.pipelines.energy_inpaint_stable_diffusion import EnergyStableDiffusionInpaintPipeline
from modules.pipelines.energy_repaint_stable_diffusion import EnergyStableDiffusionRepaintPipeline
from modules.models.energy_unet_2d_condition import EnergyUNet2DConditionModel
from modules.utils.gamma_scheduler import get_gamma_scheduler


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

def load_image(pth):
    return PIL.Image.open(pth).convert("RGB")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma_attn', type=float, default=0., help="initial weight coefficient for attention term")
    parser.add_argument('--gamma_norm', type=float, default=0., help="initial weight coefficient for normalization term")
    parser.add_argument('--token_indices', type=int, nargs='+', default=None, help="(Optional) Indices of tokens to be upweighted")
    parser.add_argument('--token_upweight', type=float, nargs='+', default=None, help="(Optional) Upweight hyperparameter")
    parser.add_argument('--num_steps', type=int, default=50, help="number of sampling steps")
    parser.add_argument('--prompt', type=str, default='teddy bear', help="text prompt")
    parser.add_argument('--img_file', type=str, default='sample/starry_night_512.png', help="img file")
    parser.add_argument('--mask_file', type=str, default='sample/starry_night_512_mask.png', help="mask file")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--repaint', action='store_true', default=False, help="If True, run SDRepaint. Otherwise, run SDInpaint")
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--get_attention', action='store_true', default=False, help="Visualize attention map. To be implemented")
    parser.add_argument('--result_file_name', type=str, default=None, help=".png result file name")
    args = parser.parse_args()

    # Prepare model
    if not args.repaint:
        model_id = "runwayml/stable-diffusion-inpainting"
    else:
        model_id = "runwayml/stable-diffusion-v1-5"

    unet = EnergyUNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        torch_dtype=torch.float16,
        down_block_types=(
            "EnergyCrossAttnDownBlock2D", "EnergyCrossAttnDownBlock2D", "EnergyCrossAttnDownBlock2D", "DownBlock2D",
        ),
        mid_block_type="EnergyUNetMidBlock2DCrossAttn",
        up_block_types=(
            "UpBlock2D", "EnergyCrossAttnUpBlock2D", "EnergyCrossAttnUpBlock2D", "EnergyCrossAttnUpBlock2D"
        ),
    )

    # Prepare gamma scheduler
    # StableRepaint resample time steps. Thus, the num_time_steps is set as arbitrarily large number
    gamma_attn = get_gamma_scheduler(name='constant', gamma_src=args.gamma_attn)(num_time_steps=1000)
    gamma_norm = get_gamma_scheduler(name='constant', gamma_src=args.gamma_norm)(num_time_steps=1000)

    # Prepare Pipeline
    if args.repaint:
        pipe = EnergyStableDiffusionRepaintPipeline.from_pretrained(model_id, unet=unet, torch_dtype=torch.float16)
        pipe.scheduler = RePaintScheduler.from_config(pipe.scheduler.config)
    else:
        pipe = EnergyStableDiffusionInpaintPipeline.from_pretrained(model_id, unet=unet, torch_dtype=torch.float16)

    pipe = pipe.to("cuda")
    generator = torch.Generator('cuda').manual_seed(args.seed)

    # Load image, mask and prompt
    init_image = load_image(f'{args.img_file}').resize((512, 512))
    mask_image = load_image(f'{args.mask_file}').resize((512, 512))
    if args.repaint:
        mask_image = PIL.ImageOps.invert(mask_image)

    # Prepare wandb
    if not args.debug:
        config = {
            'gamma_attn': args.gamma_attn,
            'gamma_norm': args.gamma_norm,
            'token_indices': args.token_indices,
            'token_upweight': args.token_upweight,
            'seed': args.seed
        }
        wandb.init(project="energy-attention",
                   entity="energy_attention",
                   tags=["inpainting", "text2img"],
                   group=args.prompt[:128],  # API limit
                   name=f"text2inpaint_gamAttn={args.gamma_attn}_gamNorm={args.gamma_norm}",
                   config=config)

    img = pipe(prompt=args.prompt,
               image=init_image,
               mask_image=mask_image,
               generator=generator,
               num_inference_steps=args.num_steps,
               gamma_attn=gamma_attn,
               gamma_norm=gamma_norm,
               token_indices=args.token_indices,
               token_upweight=args.token_upweight,
               get_attention=args.get_attention,
               ).images[0]


    if not args.debug:
        wandb.log({
            'image': wandb.Image(img, caption=args.prompt),
        })
    img.save(f'{args.result_file_name}' if args.result_file_name is not None else 'test.png')
    init_image.save('init.png')
    mask_image.save('mask.png')

if __name__ == '__main__':
    main()
