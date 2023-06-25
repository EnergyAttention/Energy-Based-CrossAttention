from typing import Union, Optional, List, Dict, Callable, Any
from itertools import repeat
import copy
import logging

import torch

from modules.utils.attention_hook import EnergyAttentionHook
from modules.models.energy_unet_2d_condition import EnergyUNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline, StableDiffusionPipelineOutput
from diffusers.models import AutoencoderKL
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.utils import logging
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class EnergySynEditStableDiffusionPipeline(StableDiffusionPipeline, EnergyAttentionHook):
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: EnergyUNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        # Additional prompt for compositional generation
        editing_prompt: Optional[Union[str, List[str]]] = None,
        # For the degree of editing.
        alpha: Optional[List[Union[List[float], List[torch.FloatTensor]]]] = None,
        # editing_direction == 1: positive editing, -1: negative_editing
        editing_direction: List[int] = [1],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        gamma_attn: Optional[List[Union[float, torch.FloatTensor]]] = None,
        gamma_norm: Optional[List[Union[float, torch.FloatTensor]]] = None,
        gamma_attn_comp: Optional[List[Union[float, torch.FloatTensor]]] = None,
        gamma_norm_comp: Optional[List[Union[float, torch.FloatTensor]]] = None,
        get_attention: Optional[bool] = False,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
        """
        if get_attention:
            self.add_hook(self.unet)

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # Encode editing prompts
        if isinstance(editing_prompt, str):
            editing_prompt = [editing_prompt]
        edit_context_inputs = self.tokenizer(
            [x for item in editing_prompt for x in repeat(item, batch_size)],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        edit_context_input_ids = edit_context_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= edit_context_input_ids.shape[-1] and not torch.equal(
            edit_context_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = edit_context_input_ids.attention_mask.to(device)
        else:
            attention_mask = None

        edit_prompt_embeds = self.text_encoder(
            edit_context_input_ids.to(device),
            attention_mask=attention_mask,
        )[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed_edit, seq_len_edit, _ = edit_prompt_embeds.shape
        edit_prompt_embeds = edit_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        edit_prompt_embeds = edit_prompt_embeds.view(bs_embed_edit * num_images_per_prompt, seq_len_edit, -1)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        # First embed of prompt_embeds corresponds to the unconditional one
        if do_classifier_free_guidance:
            edit_prompt_embeds = torch.cat([prompt_embeds[0].unsqueeze(0), edit_prompt_embeds])

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # ! ---- Give Gamma, alpha values and edit embeds as kwargs ---- #
                if cross_attention_kwargs is None:
                    cross_attention_kwargs= {}

                cross_attention_kwargs.update({
                    'gamma_attn': gamma_attn[i],
                    'gamma_attn_comp': [g[i] for g in gamma_attn_comp],
                    'gamma_norm': gamma_norm[i],
                    'gamma_norm_comp': [g[i] for g in gamma_norm_comp],
                    'alpha': [a[i] for a in alpha],
                    'edit_prompt_embeds': edit_prompt_embeds, # TODO: dict-merge with encoder_hidden_states
                    'editing_direction':editing_direction
                    })
                # ! ------------------------------------- #

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

            # 10. Convert to PIL
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
