import torch
from torch import nn
from torch.nn import functional as F
import json
import fastmri.models
from models.rec_models.unet_model import UnetModel
from models.rec_models.complex_unet import ComplexUnetModel
import data.transforms as transforms
from pytorch_nufft.nufft import nufft, nufft_adjoint
import numpy as np
from WaveformProjection.run_projection import proj_handler
import matplotlib.pylab as P
from models.rec_models.vision_transformer import VisionTransformer
from models.rec_models.recon_net import ReconNet
from models.rec_models.humus_net import HUMUSNet, HUMUSBlock
from  models.VarBlock import VarNet
from typing import Tuple
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import apply_mask
import random

import os
import torch
import numpy as np

import torch
import torchvision
import matplotlib.pyplot as plt


def save_image(source, folder_path, image_name):
    source = source.clone()
    for i in range(source.size(0)):  # Iterate over the batch dimension
        image = source[i]
        image -= image.min()
        max_val = image.max()
        if max_val > 0:
            image /= max_val
        source[i] = image

    if source.dim() == 3:
        source = source.unsqueeze(1)

    grid = torchvision.utils.make_grid(source, nrow=4, pad_value=1)
    numpy_image = grid.permute(1, 2, 0).cpu().detach().numpy()

    os.makedirs(folder_path, exist_ok=True)

    save_path = os.path.join(folder_path, f'{image_name}.png')
    plt.imsave(save_path, numpy_image)

class Subsampling_Layer(nn.Module):

    def initilaize_trajectory(self, trajectory_learning, initialization, n_shots):
        # x = torch.zeros(self.num_measurements, 2)
        sample_per_shot = self.sample_per_shot # TODO, try to increase this (instead of #of shots)
        print(sample_per_shot)
        if initialization == 'spiral':
            x = np.load(f'spiral/{n_shots}int_spiral_low.npy')
            x = torch.tensor(x[:, :sample_per_shot, :]).float()
        elif initialization == 'spiral_high':
            x = np.load(f'spiral/{n_shots}int_spiral.npy') * 10
            x = torch.tensor(x[:, :sample_per_shot, :]).float()
        elif initialization == 'EPI':
            x = torch.zeros(n_shots, sample_per_shot, 2)
            v_space = self.res // n_shots
            for i in range(n_shots):
                index = 0
                for j in range(sample_per_shot):
                    x[i, index, 1] = (i + 0.5) * v_space - 160
                    x[i, index, 0] = j * 320 / sample_per_shot - 160
                    index += 1
        elif initialization == 'radial':
            x = torch.zeros(n_shots, sample_per_shot, 2)
            theta = np.pi / n_shots
            for i in range(n_shots):
                L = torch.arange(-160, 160, 320 / sample_per_shot).float()
                x[i, :, 0] = L * np.cos(theta * i)
                x[i, :, 1] = L * np.sin(theta * i)
        elif initialization == 'uniform':
            x = (torch.rand(n_shots, sample_per_shot, 2) - 0.5) * self.res
        elif initialization == 'gaussian':
            x = torch.randn(n_shots, sample_per_shot, 2) * self.res / 6
        else:
            print('Wrong initialization')
        self.x = torch.nn.Parameter(x, requires_grad=bool(int(trajectory_learning)))
        return

    def __init__(self, decimation_rate, res, trajectory_learning, initialization, n_shots, interp_gap, SNR=False,
                 projection_iters=10, sample_per_shot = 3001, interp_gap_mode = 0, epsilon = 0, epsilon_step = None,
                 noise_type = None, noise_mode = None, noise_p = 0):
        super().__init__()

        random.seed(42)
        self.sample_per_shot = sample_per_shot
        self.decimation_rate = decimation_rate
        self.res = res
        self.num_measurements = res ** 2 // decimation_rate
        self.initilaize_trajectory(trajectory_learning, initialization, n_shots)
        self.SNR = SNR
        self.interp_gap = interp_gap
        self.iters = projection_iters
        self.epsilon = epsilon
        self.epsilon_step = epsilon_step
        self.noise_type = noise_type
        self.noise_mode = noise_mode
        self.noise_p = noise_p
        print(noise_p)

    def increase_noise_linearly(self):
        """
        Linearly increases the noise magnitude epsilon over the specified steps.

        The magnitude increases from the initial value up to the target value at the final step.
        """
        if self.epsilon == 0:
            return

        self.epsilon += self.epsilon_step
        return self.epsilon

    def add_normed_noise(self, x_full):
        """
        Adds noise to x_full with a fixed magnitude epsilon, either in L1 or L2 norm.

        Args:
            x_full (torch.Tensor): The input tensor to which noise will be added.
        Returns:
            torch.Tensor: The noisy tensor, clamped within [-160, 160].
        """
        if self.noise_p >= 0 and (not self.training or self.epsilon == 0 or random.random() <= (1 - self.noise_p)):
            return x_full
        noise = torch.randn_like(x_full) if self.noise_mode == "random" else torch.ones_like(x_full)
        if self.noise_type == "l1":
            norm = torch.norm(noise, p=1, dim=-1, keepdim=True)
        elif self.noise_type == "l2":
            norm = torch.norm(noise, p=2, dim=-1, keepdim=True)
        else:
            raise ValueError("noise_type must be 'l1' or 'l2'")

        if self.noise_p != -1:
            scaled_noise = self.epsilon * noise / (norm + 1e-8)
        else:
            scaled_noise = self.epsilon * random.random() * noise / (norm + 1e-8)

        noisy_x = x_full + scaled_noise
        noisy_x = torch.clamp(noisy_x, min=-160, max=160)
        return noisy_x

    def forward(self, input):
        if self.interp_gap > 1:
            t = torch.arange(0, self.x.shape[1], device=self.x.device).float()
            t1 = t[::self.interp_gap]
            x_short = self.x[:, ::self.interp_gap, :]
            for shot in range(x_short.shape[0]):
                for d in range(2):
                    self.x.data[shot, :, d] = self.interp(t1, x_short[shot, :, d], t)

        x_full = self.add_normed_noise(self.x.reshape(-1,2))
        input = input.permute(0, 1, 4, 2, 3)
        sub_ksp = nufft(input, x_full)
        if self.SNR:
            noise_amp = 0.001
            noise = noise_amp * torch.randn(sub_ksp.shape)
            sub_ksp = sub_ksp + noise.to(sub_ksp.device)
        output = nufft_adjoint(sub_ksp, x_full, input.shape)
        output = output.permute(0, 1, 3, 4, 2)
        return output

    def get_trajectory(self):
        return self.x

    def h_poly(self, t):
        tt = [None for _ in range(4)]
        tt[0] = 1
        for i in range(1, 4):
            tt[i] = tt[i - 1] * t
        A = torch.tensor([
            [1, 0, -3, 2],
            [0, 1, -2, 1],
            [0, 0, 3, -2],
            [0, 0, -1, 1]
        ], dtype=tt[-1].dtype)
        return [
            sum(A[i, j] * tt[j] for j in range(4))
            for i in range(4)]

    def interp(self, x, y, xs):
        m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
        m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
        I = P.searchsorted(x[1:].detach().cpu(), xs.detach().cpu())
        dx = (x[I + 1] - x[I])
        hh = self.h_poly((xs - x[I]) / dx)
        return hh[0] * y[I] + hh[1] * m[I] * dx + hh[2] * y[I + 1] + hh[3] * m[I + 1] * dx

    def trajectory_to_mask(
            self,
            grid_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Converts a k-space trajectory into a binary mask.

        Args:
            trajectory (torch.Tensor): A tensor of shape (-1, 2) representing the k-space trajectory.
                                       Each row contains (kx, ky) coordinates normalized to [-1, 1].
            grid_shape (Tuple[int, int]): The shape of the mask grid (height, width).

        Returns:
            torch.Tensor: A binary mask of shape (height, width) with 1s at sampled locations.
        """
        H, W = grid_shape  # Unpack grid dimensions

        # Normalize trajectory to grid coordinates
        trajectory = self.x.reshape(-1,2)
        print(trajectory)
        grid_kx = ((trajectory[:, 0] + 1) / 2 * (W - 1)).long()
        grid_ky = ((trajectory[:, 1] + 1) / 2 * (H - 1)).long()

        # Clip to ensure indices are within bounds
        grid_kx = grid_kx.clamp(0, W - 1)
        grid_ky = grid_ky.clamp(0, H - 1)

        # Initialize an empty mask
        mask = torch.zeros(H, W, dtype=torch.int8)

        # Set sampled locations to 1
        mask[grid_ky, grid_kx] = 1

        print(mask)
        return mask

    def __repr__(self):
        return f'Subsampling_Layer'

class HUMUSReconstructionModel(nn.Module):
    def __init__(self, img_size, in_chans, out_chans, num_blocks, window_size, embed_dim):
        super(HUMUSReconstructionModel, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                HUMUSBlock(
                    img_size=img_size,
                    in_chans=in_chans if i == 0 else out_chans,
                    out_chans=out_chans,
                    window_size=window_size,
                    embed_dim=embed_dim
                )
            )
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class Subsampling_Model(nn.Module):
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob, decimation_rate, res,
                 trajectory_learning, initialization, n_shots, interp_gap, SNR=False, type=None,
                 img_size=(320, 320) ,window_size=10, embed_dim=66, num_blocks=1, sample_per_shot=3001,
                 epsilon = 0, epsilon_step = None, noise_type = None, noise_mode = None, noise_p = 0):
        super().__init__()
        self.type = type
        self.subsampling = Subsampling_Layer(decimation_rate, res, trajectory_learning, initialization, n_shots,
                                             interp_gap, SNR, sample_per_shot = sample_per_shot, epsilon = epsilon,
                                             epsilon_step = epsilon_step, noise_type = noise_type, noise_mode = noise_mode,
                                             noise_p = noise_p)
        if type == 'vit':
            avrg_img_size = 320
            patch_size = 10
            depth = 4
            num_heads = 9
            embed_dim = 300
            vitNet = VisionTransformer(
                avrg_img_size=avrg_img_size,
                patch_size=patch_size,
                in_chans=1, embed_dim=embed_dim,
                depth=depth, num_heads=num_heads,
            )
            self.reconstruction_model = vitNet
        elif type == 'Unet':
            self.reconstruction_model = UnetModel(in_chans, out_chans, chans, num_pool_layers, drop_prob)
        else:
            HUMUSReconstructionNet = HUMUSBlock(img_size=img_size, in_chans =in_chans, out_chans = out_chans, num_blocks = num_blocks, window_size = window_size, embed_dim = embed_dim)
            self.reconstruction_model = HUMUSReconstructionNet

        self.iter = 0


    def forward(self, input):
        if self.type == "vit":
            input = self.subsample_and_stack(input).squeeze()
            kspace = transforms.fft2(input)
            mask = self.subsampling.trajectory_to_mask((kspace.shape[-3], kspace.shape[-2]))
            print(mask.shape)
            num_ones = torch.sum(mask == 1).item()
            input = transforms.root_sum_of_squares(transforms.complex_abs(input), dim=0).unsqueeze(0).unsqueeze(0)
            return self.reconstruction_model(input, None)
        elif self.type == 'Unet':
            input = self.subsampling(input)
            input = transforms.root_sum_of_squares(transforms.complex_abs(input), dim=1).unsqueeze(1)
            output = self.reconstruction_model(input).reshape(-1, 320, 320)
            return output
        elif self.type == "humus":
            input = self.subsampling(input).squeeze(0).permute(0,3,1,2)
            out = self.reconstruction_model(input)
            out = out.permute(0,2,3,1)
            output = transforms.root_sum_of_squares(transforms.complex_abs(out.unsqueeze(0)), dim=1)
            return output.reshape(-1, 320, 320)
        return None

    def get_trajectory(self):
        return self.subsampling.get_trajectory()

    def increase_noise_linearly(self):
        self.subsampling.increase_noise_linearly()