import argparse
import torch
import wandb
from modules.pipelines.energy_realedit_stable_diffusion import EnergyRealEditStableDiffusionPipeline
from modules.models.energy_unet_2d_condition import EnergyUNet2DConditionModel
from modules.utils.gamma_scheduler import get_gamma_scheduler

from transformers import BlipForConditionalGeneration, BlipProcessor
from diffusers import DDIMScheduler, DDIMInverseScheduler
import requests
from PIL import Image
from inpaint_txt2img import load_image, download_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma_attn', type=float, default=0., help="initial weight coefficient for attention term")
    parser.add_argument('--gamma_norm', type=float, default=0., help="initial weight coefficient for normalization term")
    parser.add_argument('--num_steps', type=int, default=50, help="number of sampling steps")
    parser.add_argument('--prompt', type=str, default=None, help="main prompt")
    parser.add_argument('--editing_prompt', type=str, nargs='+', default='', help="additional prompt for composition")
    parser.add_argument('--alpha', type=float, default=0., nargs='+', help="non-negative degree of additional composition.")
    parser.add_argument('--alpha_tau', type=float, default=0., nargs='+', help="Turn on alpha after some threshold time. r == 0: alpha=0")
    parser.add_argument('--gamma_attn_compose', type=float, default=0., nargs='+', help="non-negative degree of editorial context update.")
    parser.add_argument('--gamma_norm_compose', type=float, default=0., nargs='+', help="non-negative degree of editorial context update.")
    parser.add_argument('--gamma_tau', type=float, default=0., nargs='+', help="Turn on gamma_compose after some time (like alpha_tau).")
    parser.add_argument('--editing_direction', type=int, nargs='+', default=1, choices={0, 1}, help="If 1, positive composition. Elif 0, negative composition")
    parser.add_argument('--negative_prompt', type=str, default='', help="negative prompt")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--v4', action='store_true', default=False)
    parser.add_argument('--img_file', type=str, default='sample/starry_night_512.png', help="img file")
    parser.add_argument('--save_name', type=str, default=None, help="img file")
    parser.add_argument('--get_attention', action='store_true', default=False)
    args = parser.parse_args()

    # Prepare model
    if args.v4:
        model_id = "CompVis/stable-diffusion-v1-4"
    else:
        # v5 is default
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
    gamma_attn = get_gamma_scheduler(name='constant', gamma_src=args.gamma_attn)(num_time_steps=args.num_steps)
    gamma_norm = get_gamma_scheduler(name='constant', gamma_src=args.gamma_norm)(num_time_steps=args.num_steps)

    # Prepare composition & alpha scheduler
    if isinstance(args.editing_prompt, str):
        args.editing_prompt = [args.editing_prompt]
    if isinstance(args.alpha, float):
        args.alpha = [args.alpha]
    if isinstance(args.alpha_tau, float):
        args.alpha_tau = [args.alpha_tau]
    if isinstance(args.gamma_attn_compose, float):
        args.gamma_attn_compose = [args.gamma_attn_compose]
    if isinstance(args.gamma_norm_compose, float):
        args.gamma_norm_compose = [args.gamma_norm_compose]
    if isinstance(args.gamma_tau, float):
        args.gamma_tau = [args.gamma_tau]
    if isinstance(args.editing_direction, int):
        args.editing_direction = [args.editing_direction]

    alpha_warm = [
        get_gamma_scheduler(name='step', gamma_src=args.alpha[i], gamma_tau=args.alpha_tau[i])(num_time_steps=args.num_steps) \
        for i in range(len(args.alpha))
    ]

    gamma_attn_compose_warm = [
        get_gamma_scheduler(name='step', gamma_src=args.gamma_attn_compose[i], gamma_tau=args.gamma_tau[i])(num_time_steps=args.num_steps) \
        for i in range(len(args.alpha))
    ]
    gamma_norm_compose_warm = [
        get_gamma_scheduler(name='step', gamma_src=args.gamma_norm_compose[i], gamma_tau=args.gamma_tau[i])(num_time_steps=args.num_steps) \
        for i in range(len(args.alpha))
    ]

    # Prepare BLIP for captioning
    captioner_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(captioner_id)
    model = BlipForConditionalGeneration.from_pretrained(
        captioner_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )

    # Prepare Pipeline
    pipe = EnergyRealEditStableDiffusionPipeline.from_pretrained(model_id,
                                                                 unet=unet,
                                                                 caption_generator=model,
                                                                 caption_processor=processor,
                                                                 torch_dtype=torch.float16,
                                                                 safety_checker=None
                                                                 )

    # Prepare inversion scheduler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    #pipe.enable_model_cpu_offload()

    pipe = pipe.to("cuda")
    generator = torch.Generator('cuda').manual_seed(args.seed)

    # ! ------------------------------------------------------------------------
    # Obtain inverted latents for a given image
    raw_image = load_image(f'{args.img_file}').resize((512, 512))
    raw_image.save('original.jpg')

    # generate caption
    caption = pipe.generate_caption(raw_image)
    print('caption: ', caption)

    # "a photography of a cat with flowers and dai dai daie - daie - daie kasaii"
    inv_latents = pipe.invert(caption, image=raw_image).latents
    # ! ------------------------------------------------------------------------

    # Prepare wandb
    if not args.debug:
        concat = lambda values: ','.join([str(x) for x in values])
        config = {
            'alpha': concat(args.alpha),
            'alpha_tau': concat(args.alpha_tau),
            'gamma_attn': args.gamma_attn,
            'gamma_norm': args.gamma_norm,
            'gamma_attn_compose': args.gamma_attn_compose,
            'gamma_norm_compose': args.gamma_norm_compose,
            'seed': args.seed,
            'editing_prompt': args.editing_prompt,
            'editing_direction': concat(args.editing_direction)
        }
        wandb.init(project="energy-attention",
                   entity="energy_attention",
                   tags=["editing", "text2img"],
                   group=caption,  # API limit
                   name=f"text2img_alpha={args.alpha[0]}_alphaR={args.alpha_tau[0]}",
                   config=config)

    img = pipe(args.prompt if args.prompt is not None else caption,
               generator=generator,
               latents=inv_latents,
               negative_prompt=caption,
               num_inference_steps=args.num_steps,
               gamma_attn=gamma_attn,
               gamma_norm=gamma_norm,
               editing_prompt=args.editing_prompt,
               alpha=alpha_warm,
               gamma_attn_comp=gamma_attn_compose_warm,
               gamma_norm_comp=gamma_norm_compose_warm,
               editing_direction=args.editing_direction,
               get_attention=args.get_attention,
               ).images[0]

    if not args.debug:
        prompts = [args.prompt] + args.editing_prompt
        wandb.log({
            'image': wandb.Image(img, caption=' |'.join([caption] + args.editing_prompt)),
        })

    img.save(args.save_name if args.save_name is not None else 'test.png')

if __name__ == '__main__':
    main()
