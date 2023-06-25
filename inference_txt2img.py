import argparse
import torch
import wandb
from modules.pipelines.energy_stable_diffusion import EnergyStableDiffusionPipeline
from modules.models.energy_unet_2d_condition import EnergyUNet2DConditionModel
from modules.utils.gamma_scheduler import get_gamma_scheduler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma_attn', type=float, default=0., help="initial weight coefficient for attention term")
    parser.add_argument('--gamma_norm', type=float, default=0., help="initial weight coefficient for normalization term")
    parser.add_argument('--gamma_tau', type=float, default=1., help="Turn off gammas after some time. r == 1: gamma never turns off.")
    parser.add_argument('--token_indices', type=int, nargs='+', default=None, help="(Optional) Indices of tokens to be upweighted")
    parser.add_argument('--token_upweight', type=float, nargs='+', default=None, help="(Optional) Upweight hyperparameter")
    parser.add_argument('--num_steps', type=int, default=50, help="number of sampling steps")
    parser.add_argument('--prompt', type=str, default='A standing dog', help="text prompt")
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--seed', type=int, nargs='+', default=0, help="random seed")
    parser.add_argument('--v4', action='store_true', default=False)
    parser.add_argument('--result_file_name', type=str, default='', help=".png result file name")
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
    gamma_attn = get_gamma_scheduler(name='reverse_step',
                                     gamma_tau=args.gamma_tau, gamma_src=args.gamma_attn)(num_time_steps=args.num_steps)
    gamma_norm = get_gamma_scheduler(name='reverse_step',
                                     gamma_tau=args.gamma_tau, gamma_src=args.gamma_norm)(num_time_steps=args.num_steps)

    # Prepare Pipeline
    pipe = EnergyStableDiffusionPipeline.from_pretrained(model_id, unet=unet, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    for seed in args.seed:
        generator = torch.Generator('cuda').manual_seed(seed)

        # Prepare wandb
        if not args.debug:
            config = {
                'gamma_attn': args.gamma_attn,
                'gamma_norm': args.gamma_norm,
                'gamma_tau': args.gamma_tau,
                'token_indices': args.token_indices,
                'token_upweight': args.token_upweight,
                'seed': seed
            }
            wandb.init(project="energy-attention",
                    entity="energy_attention",
                    tags=["ablation", "generation", "multi-concept", "text2img"],
                    group=args.prompt[:128],  # API limit
                    name=f"text2img_gamAttn={args.gamma_attn}_gamNorm={args.gamma_norm}_gamRatio={args.gamma_tau}",
                    config=config)

        img = pipe(args.prompt,
                   generator=generator,
                   num_inference_steps=args.num_steps,
                   gamma_attn=gamma_attn,
                   gamma_norm=gamma_norm,
                   token_indices=args.token_indices,
                   token_upweight=args.token_upweight,
                   get_attention=args.get_attention).images[0]

        if not args.debug:
            wandb.log({
                'image': wandb.Image(img, caption=args.prompt),
            })
        img.save(f'{args.result_file_name}seed{seed}.png')

if __name__ == '__main__':
    main()
