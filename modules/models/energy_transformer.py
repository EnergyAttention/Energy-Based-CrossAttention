from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers.models.attention import BasicTransformerBlock
from .energy_attention import EnergyCrossAttention
import torch


class EnergyTransformerBlock(BasicTransformerBlock):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
    ):
        super().__init__(dim=dim,
                         num_attention_heads=num_attention_heads,
                         attention_head_dim=attention_head_dim,
                         dropout=dropout,
                         cross_attention_dim=cross_attention_dim,
                         activation_fn=activation_fn,
                         num_embeds_ada_norm=num_embeds_ada_norm,
                         attention_bias=attention_bias,
                         only_cross_attention=only_cross_attention,
                         upcast_attention=upcast_attention,
                         norm_elementwise_affine=norm_elementwise_affine,
                         norm_type=norm_type,
                         final_dropout=final_dropout)

        # 1. Self-Attn
        self.attn1 = EnergyCrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None:
            self.attn2 = EnergyCrossAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.attn2 = None

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        attention_mask=None,
        cross_attention_kwargs=None,
        class_labels=None,
    ):
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        # 1. Self-Attention
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        attn_output, _, _, _ = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )

            # 2. Cross-Attention
            attn_output, update_c, update_edit, _ = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

            # ------------------------------------------------------------------
            # Energy-based Bayesian context update
            # (1) main context
            if encoder_hidden_states is not None:
                update_c = encoder_hidden_states + update_c

            # (2, optional) editorial contexts
            if 'edit_prompt_embeds' in cross_attention_kwargs and len(update_edit) != 0:
                self.update_editorial_contexts(update_edit, cross_attention_kwargs)
            # ------------------------------------------------------------------

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states, update_c

    def update_editorial_contexts(self,
                                  update_edit: List[torch.Tensor],
                                  cross_attention_kwargs):
        """
        Bayesian Context Update (BCU) of editorial embeddings.
        For real and synthetic image editing, BCU can be applied to both main and editorial embeddings.
        Here, the editorial contexts are updated in-place within cross_attention_kwargs dictionary.
        Note that the (optimized) unconditional textual embedding vector is also targeted for additional BCUs.
        For the update of unconditional embedding, update terms of each editorial embedding are averaged out.

        update_edit: List of [2, N, D]-sized embeddings. 2: uncond & cond embeddings, N: # of tokens, D: embedding dimension
        cross_attention_kwargs: contains edit_prompt_embeds that are to be updated
        """
        N = len(update_edit)
        for i, update_edit_i in enumerate(update_edit):
            # Update conditional part
            cross_attention_kwargs['edit_prompt_embeds'][i+1] += update_edit_i[1]
            # Update unconditional part
            cross_attention_kwargs['edit_prompt_embeds'][0] += update_edit_i[0] / N


