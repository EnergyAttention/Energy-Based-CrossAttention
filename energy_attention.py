from math import sqrt
from typing import Optional, List
import torch
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import Attention


class EnergyCrossAttnProcessor:
    def __call__(
        self,
        attn: Attention,
        hidden_states,
        gamma_attn: float=0.,
        gamma_norm: float=0.,
        encoder_hidden_states=None,
        attention_mask=None,
        # For text-guided inpainting
        downsample_mask_fn=None,
        # For compositional editing
        alpha: List[float]=[0.],
        edit_prompt_embeds: Optional[List[torch.Tensor]]=None,
        editing_direction: List[int]=[False],
        gamma_attn_comp: List[float]=[0.,],
        gamma_norm_comp: List[float]=[0.,],
        # For token-wise gamma_attn tuning
        token_indices: Optional[List[int]]=None,
        token_upweight: Optional[List[float]]=None,
    ):
        """
        Energy-based Cross-Attention (EBCA) implementation.
        While the conventional cross-attention only takes hidden states (for q) and encoder_hidden_states (for k,v),
        the proposed EBCA takes hyper-parameters, editorial embeddings, etc, as additional inputs.
        Moreover, EBCA additionally returns C_main and C_edit which are update terms of main and editorial embeddings.
        Both C_main and C_edit are computed by Bayesian Context Update (BCU).
        These update terms are added to the original embeddings in energy_transformer.py.
        For CACAO, edit_prompt_embeds serve as additional editorial embeddings.

        [Introduced hyper-parameters]
        - gamma_attn, gamma_norm: Step sizes of BCU of main embedding
        - downsample_mask_fn: For text-guided inpainting. lambda function which takes the spatial resolution of the corresponding layer
            and returns the downsampled inpainting mask.
        - alpha: linear coefficient of CACAO
        - edit_prompt_embeds: list of editorial prompt embeddings
        - editing_direction: indicating conjunction/negation of compositional editing. +1: conjunction, else: negation
        - gamma_attn_comp, gamma_norm_comp: Step sizes of BCU of each editorial embedding
        - token_indices: indices of textual token for token-wise gamma_attn, e.g. [2,5] indicates
        2nd and 5th token.
        - token_upweight: increases i-th gamma_attn of main textual embedding following
        token_indices. Token-wise gamma_attn tuning is not applied to the gamma_attn_comp because
        edit_prompt_embeds are encoded from the whole editorial prompts.
        """
        # batch_size = 2*num_of_images_wewant_to_generate (due to cfg)
        batch_size, sequence_length, _ = hidden_states.shape
        N = batch_size // 2
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        self_attention = False

        if encoder_hidden_states is None:
            self_attention = True
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        output_state_temp, query, key, attention_scores = self.attend(hidden_states, encoder_hidden_states,
                                                                      attn, attention_mask)

        # ----------------------------------------------------------------------
        # For composition:
        # Additional editting for compositional generation
        # ----------------------------------------------------------------------
        if edit_prompt_embeds is not None and not self_attention:
            # C_edit: Bayesian context update for editorial embeddings
            output_state, C_edit = self.cacao(
                output_state_temp,
                edit_prompt_embeds,
                editing_direction,
                hidden_states,
                attn,
                attention_mask,
                N,
                alpha,
                gamma_attn_comp,
                gamma_norm_comp,
            )
        else:
            output_state = output_state_temp
            C_edit = []
        # ----------------------------------------------------------------------

        # C_main: Bayesian context update for main embeddings
        C_main = self.bayesian_context_update(
            attention_scores,
            query,
            key,
            gamma_attn,
            gamma_norm,
            attn,
            N,
            downsample_mask_fn=downsample_mask_fn,
            token_indices=token_indices,
            token_upweight=token_upweight
        )

        return output_state, C_main, C_edit, attention_scores

    def cacao(
        self,
        output_state_temp: torch.Tensor,
        edit_prompt_embeds: List[torch.Tensor],
        editing_direction: List[int],
        hidden_states: torch.Tensor,
        attn: Attention,
        attention_mask,
        N: int,
        alpha: List[float],
        gamma_attn_comp: List[float],
        gamma_norm_comp: List[float],
    ):
        C_edit = []

        # Let N=#number_of_images_wewant_to_generate. Then,
        # len(edit_prompt_embeds) = N + num_edit_prompt*N, where first N corresponds to the
        # unconditional embeds.
        num_edit_prompt = (len(edit_prompt_embeds) - N) // N
        assert len(alpha) == num_edit_prompt
        uncond_embed = edit_prompt_embeds[:N]
        edit_strength = 1.

        for i, edit_prompt_embed in enumerate(edit_prompt_embeds[N:].chunk(num_edit_prompt)):
            # 1. cross-attention map for each editorial context
            edit_prompt_embed_with_uncond = torch.cat((uncond_embed, edit_prompt_embed))
            output_state_edit, query_edit, key_edit, attention_scores_edit = \
                self.attend(hidden_states, edit_prompt_embed_with_uncond, attn, attention_mask)

            # 2. Bayesian context update for editorial embeddings
            # C_edit_i contains update term of both uncond. and cond. embeddings.
            C_edit_i = self.bayesian_context_update(
                attention_scores_edit,
                query_edit,
                key_edit,
                gamma_attn_comp[i],
                gamma_norm_comp[i],
                attn,
                N
            )
            C_edit.append(C_edit_i)

            sgn = 1. if editing_direction[i] == 1 else -1.
            output_state_temp = output_state_temp + sgn * alpha[i] * output_state_edit
            edit_strength += sgn * alpha[i]
        output_state = output_state_temp / edit_strength
        return output_state, C_edit

    def bayesian_context_update(
        self,
        attention_scores: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        gamma_attn: float,
        gamma_norm: float,
        attn: Attention,
        N: int,
        downsample_mask_fn=None,
        token_indices: Optional[List[int]]=None,
        token_upweight: Optional[List[float]]=None,
    ):
        # calculate att-prob for context
        # TODO: forward update of edit_prompt_embeds if exists
        attention_probs = attention_scores.mT.softmax(dim=-1)
        attention_probs = attention_probs.to(query.dtype)

        # ----------------------------------------------------------------------
        # For text-guided inpainting:
        # downsample_mask_fn: default: None
        # ----------------------------------------------------------------------
        if downsample_mask_fn is not None:
            query_spatial_dim = int(sqrt(query.shape[-2]))
            mask_ = downsample_mask_fn(query_spatial_dim).reshape(-1, 1)
            C = torch.bmm(attention_probs, (query*mask_))
        else:
            C = torch.bmm(attention_probs, query)
        C = attn.batch_to_head_dim(C) * gamma_attn

        # ----------------------------------------------------------------------
        # For token-wise gamma_attn tuning
        # ----------------------------------------------------------------------
        if token_indices is not None and token_upweight is not None:
            assert len(token_indices) == len(token_upweight), \
                "Upweighting hyper-parameter is not specified correctly."
            for i, ind in enumerate(token_indices):
                C[N:, ind, :] *= token_upweight[i]
        # ----------------------------------------------------------------------

        # norm
        k = attn.batch_to_head_dim(key)
        weight = torch.diagonal(torch.matmul(k, k.mT), dim1=-2, dim2=-1)
        C -= k * weight.softmax(dim=-1).unsqueeze(-1) * gamma_norm
        C  = torch.matmul(C, attn.to_k.weight.detach())

        # C: gradient term of the update rule
        return C

    def attend(self, input_state, context_embed, attn, attention_mask):
        query = attn.to_q(input_state)
        key = attn.to_k(context_embed)
        value = attn.to_v(context_embed)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if attn.upcast_attention:
            query = query.float()
            key = key.float()

        # Attention in diffusers==0.16.1 only uses attention_probs. Thus raw_attention_scores
        # are extracted with the below function.
        attention_scores = self.get_raw_attention_scores(query, key, attn, attention_mask)
        if attn.upcast_softmax:
            attention_scores = attention_scores.float()

        # calculate att-prob for value
        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(query.dtype)
        output_state = torch.bmm(attention_probs, value)
        output_state = attn.batch_to_head_dim(output_state)

        # linear proj
        output_state = attn.to_out[0](output_state)
        # dropout
        output_state = attn.to_out[1](output_state)
        return output_state, query, key, attention_scores

    def get_raw_attention_scores(self, query, key, attn, attention_mask=None):
        dtype = query.dtype

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=attn.scale,
        )

        if attn.upcast_softmax:
            attention_scores = attention_scores.float()
        return attention_scores

class EnergyCrossAttention(Attention):
    def __init__(
            self,
            query_dim: int,
            cross_attention_dim: Optional[int] = None,
            heads: int = 8,
            dim_head: int = 64,
            dropout: float = 0.0,
            bias=False,
            upcast_attention: bool = False,
            upcast_softmax: bool = False,
            cross_attention_norm: Optional[str] = None,
            cross_attention_norm_num_groups: int = 32,
            added_kv_proj_dim: Optional[int] = None,
            norm_num_groups: Optional[int] = None,
            out_bias: bool = True,
            scale_qk: bool = True,
            only_cross_attention: bool = False,
            processor=EnergyCrossAttnProcessor,
        ):
        super().__init__(query_dim=query_dim,
                         cross_attention_dim=cross_attention_dim,
                         heads=heads,
                         dim_head=dim_head,
                         dropout=dropout,
                         bias=bias,
                         upcast_attention=upcast_attention,
                         upcast_softmax=upcast_softmax,
                         cross_attention_norm=cross_attention_norm,
                         cross_attention_norm_num_groups=cross_attention_norm_num_groups,
                         added_kv_proj_dim=added_kv_proj_dim,
                         norm_num_groups=norm_num_groups,
                         out_bias=out_bias,
                         scale_qk=scale_qk,
                         only_cross_attention=only_cross_attention,
                         processor=processor()
                    )

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        # We pass hyperparams (e.g., gammas) with `**cross_attention_kwargs`
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
