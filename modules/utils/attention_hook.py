from typing import Tuple, Union
from pathlib import Path
import math

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image

from modules.models.energy_attention import EnergyCrossAttention

class EnergyAttentionHook(object):
    def __init__(self):
        self.attn_maps = []
        self.cross_attn_num = 0
        self.sampled_step = 0
        self.head_index = 8
        self.token_index = 5
        self.layer_index = [11]

    def energy_hook(self,
                    module:nn.Module,
                    input: Union[Tuple[torch.Tensor,], torch.Tensor],
                    output: Union[Tuple[torch.Tensor,], torch.Tensor],
                    ) -> None:
        if isinstance(module, EnergyCrossAttention):
            q, c, (attn_score, k) = output
            Ecq = self.cal_energy_c_given_q(attn_score)
            Ec = self.cal_energy_c(k)
            Eqc = self.cal_energy_q_given_c(q, attn_score)

            self.add_energy(Ecq, Ec, Eqc)

    def cal_energy_c_given_q(self,
                             attn_score: torch.Tensor) -> torch.Tensor:
        return -torch.logsumexp(attn_score.mT, dim=-1)

    def cal_energy_c(self,
                     c: torch.Tensor,
                     ) -> torch.Tensor:
        c = c[-1,...]  # get conditional part
        return torch.logsumexp(0.5 * torch.diagonal(torch.matmul(c, c.mT)), dim=-1)

    def cal_energy_q_given_c(self,
                             q: torch.Tensor,
                             attn_score: torch.Tensor) -> torch.Tensor:
        q = q[-1,...]  # get conditional part
        return 0.5 * torch.diagonal(torch.matmul(q, q.mT)) - torch.logsumexp(attn_score[self.head_index,...], dim=-1)

    def add_energy(self,
                   ecq: torch.Tensor,
                   ec : torch.Tensor,
                   eqc: torch.Tensor
                  ):
        # (attn_head_num, text_num) -> (1,)
        # ecq = ecq[self.head_index].sum()
        ecq = ecq.mean(dim=0).sum()
        # (text_num) -> (1,)
        ec  = ec.sum()
        # (d_h * d_w) -> (d_h, d_w)
        latent_dim = int(math.sqrt(eqc.shape[-1]))
        assert latent_dim ** 2 == eqc.shape[-1]
        eqc = eqc.reshape(latent_dim, latent_dim)

        # write as txt file
        self.save_energy_value(ecq.detach().cpu().numpy(), 'Ecq.txt')
        self.save_energy_value(ecq.detach().cpu().numpy() + ec.detach().cpu().numpy(), 'E.txt')
        self.save_energy_value(ec.detach().cpu().numpy(), 'Ec.txt')
        # self.save_energy_value(eqc.detach().cpu().numpy(), 'Eqc.txt')

    def save_energy_value(self,
                          energy: np.ndarray,
                          fname: str) -> None:
        save_dir = Path('energy_progress')
        save_dir.mkdir(exist_ok=True)

        with open(save_dir.joinpath(fname), 'a') as f:
            f.write(f'{float(energy)}\n')

    def attn_hook(self,
                  module:nn.Module,
                  input: Union[Tuple[torch.Tensor,], torch.Tensor],
                  output: Union[Tuple[torch.Tensor,], torch.Tensor],
                  ) -> None:
        if isinstance(module, EnergyCrossAttention):
            _, _, (attn_score, _) = output
            self.add_attn_map(attn_score)

    def add_attn_map(self, attn_score: torch.Tensor) -> torch.Tensor:
        attn_score = attn_score[8,...]  # get conditional part & first head
        attn_map = attn_score.softmax(dim=-2)[:, self.token_index]  # get attn map for a token

        if len(self.attn_maps) < self.cross_attn_num:
            self.attn_maps.append(attn_map.detach().cpu())
        else:
            # save previous maps & re-init
            if (self.sampled_step+1) % 10 == 0 or self.sampled_step==0:
                self.save_attn_map()
            self.attn_maps = [attn_map.detach().cpu()]
            self.sampled_step += 1

    def save_attn_map(self):
        target_map = [self.attn_maps[x] for x in self.layer_index]
        target_map = torch.stack(target_map, dim=0).to(torch.float32)

        map_size = int(math.sqrt(target_map.shape[-1]))
        assert map_size**2 == target_map.shape[-1]

        target_map = target_map.reshape(-1, map_size, map_size)
        target_map = target_map.clamp(0, 0.005)
        save_dir = Path('attn_map')
        save_dir.mkdir(exist_ok=True)
        save_image(target_map, save_dir.joinpath(f'{self.sampled_step}.png'), nrow=len(self.layer_index), normalize=True)

    def add_hook(self, net):
        for name, layer in net._modules.items():
            if not isinstance(layer, EnergyCrossAttention):
                self.add_hook(layer)
            else:
                if not name == 'attn1':  # skip self-attn
                    layer.register_forward_hook(self.attn_hook)
                    layer.register_forward_hook(self.energy_hook)
                    self.cross_attn_num += 1

