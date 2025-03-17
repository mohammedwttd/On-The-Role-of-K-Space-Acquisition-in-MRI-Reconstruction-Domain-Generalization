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

def compute_dc_loss(y_full_kspace, x_full, kspace_values, d_N):
    """
    Computes data consistency (DC) loss using extracted values from y_full_kspace.
    Avoids direct subtraction to work around potential device/contiguity issues.
    """
    batch, coils, height, width, complex_dim = y_full_kspace.shape
    assert complex_dim == 2, "Expected real and imaginary components in last dimension."

    # Normalize coordinates
    x_full_normalized = ((x_full + 160) / 320) * (height - 1)
    x_rounded = torch.floor(x_full_normalized).long().clamp(0, height - 1)

    # Extract values from y_full_kspace
    real_vals = y_full_kspace[0, 0, x_rounded[:, 0], x_rounded[:, 1], 0]
    imag_vals = y_full_kspace[0, 0, x_rounded[:, 0], x_rounded[:, 1], 1]

    # Construct y_kspace_values and ensure contiguous memory
    y_kspace_values = torch.stack([real_vals, imag_vals], dim=0)
    y_kspace_values = y_kspace_values.unsqueeze(0).unsqueeze(0).contiguous()  # [1, 1, 2, 48016]

    # Ensure device consistency
    device = kspace_values.device
    y_kspace_values = y_kspace_values.to(device=device, dtype=kspace_values.dtype)
    #d_N = d_N.to(device=device, dtype=kspace_values.dtype)
    kspace_values = kspace_values.contiguous()

    dc_loss = F.l1_loss(y_kspace_values, kspace_values)

    #print(f"DC Loss: {dc_loss.item()}")
    return dc_loss




def compute_dc_factors(input, x_full, num_iterations=10):
    """
    Compute data consistency (DC) factors iteratively and return DC loss.

    Args:
        input: The original measurement y (in image space) (batch_size, coils, height, width, 2)
        x_full: The corresponding k-space sampling trajectory Ω.
        num_iterations: Number of iterations for convergence.

    Returns:
        d_N: Final data consistency factors (same shape as input).
        dc_loss: A scalar tensor representing the DC loss in k-space.
    """
    # Initialize d_0 = 1
    d_N = torch.ones_like(input)

    #for _ in range(num_iterations):
    #    sub_ksp = nufft(d_N, x_full)  # Forward NUFFT (compute k-space)
    #    y_k = nufft_adjoint(sub_ksp, x_full, input.shape)  # Reconstruct to image space
    #    d_N = d_N * (y_k / d_N)  # Update DC factors

    y_full_kspace = transforms.fft2(input.permute(0, 1, 3, 4, 2))
    y_kspace = nufft(input, x_full)
    # Compute DC loss in k-space: || d_N * (y_kspace - y_k) ||_2
    dc_loss = compute_dc_loss(y_full_kspace, x_full, y_kspace, d_N)

    return dc_loss


class Subsampling_Layer(nn.Module):
    def create_k_space_mask(self, k_space_size=320):
        """
        Creates a k-space mask based on a given trajectory.

        Parameters:
            k_space_size (int): The size of the k-space grid (default 320).

        Returns:
            k_space_mask (torch.Tensor): A binary mask representing sampled locations in k-space.
        """
        #print(self.x)
        n_shots, sample_per_shot, _ = self.x.shape  # Get the number of shots and samples per shot

        # Initialize the k-space mask with zeros
        k_space_mask = torch.zeros(k_space_size, k_space_size)

        # Normalize the trajectory points to fit within the k-space grid size
        for i in range(n_shots):
            for j in range(sample_per_shot):
                kx, ky = self.x[i][j][0], self.x[i][j][1]

                # Map the continuous k-space coordinates to grid indices
                kx_idx = int((kx + 160) * (k_space_size / 320))  # Normalize to grid size
                ky_idx = int((ky + 160) * (k_space_size / 320))  # Normalize to grid size

                # Ensure the indices are within the bounds of the k-space grid
                kx_idx = max(0, min(k_space_size - 1, kx_idx))
                ky_idx = max(0, min(k_space_size - 1, ky_idx))

                # Set the corresponding position in the k-space mask to 1 (sampled)
                k_space_mask[kx_idx, ky_idx] = 1

        return k_space_mask

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
                 projection_iters=10, sample_per_shot = 3001, interp_gap_mode = 0):
        super().__init__()

        self.sample_per_shot = sample_per_shot
        self.decimation_rate = decimation_rate
        self.res = res
        self.num_measurements = res ** 2 // decimation_rate
        self.initilaize_trajectory(trajectory_learning, initialization, n_shots)
        self.SNR = SNR
        self.interp_gap = interp_gap
        self.iters = projection_iters

    def forward(self, input):
        if self.interp_gap > 1:
            t = torch.arange(0, self.x.shape[1], device=self.x.device).float()
            t1 = t[::self.interp_gap]
            x_short = self.x[:, ::self.interp_gap, :]
            for shot in range(x_short.shape[0]):
                for d in range(2):
                    self.x.data[shot, :, d] = self.interp(t1, x_short[shot, :, d], t)

        x_full = self.x.reshape(-1,2)
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
                 img_size=(320, 320) ,window_size=10, embed_dim=66, num_blocks=1, sample_per_shot=3001):
        super().__init__()
        self.type = type
        self.subsampling = Subsampling_Layer(decimation_rate, res, trajectory_learning, initialization, n_shots,
                                             interp_gap, SNR, sample_per_shot = sample_per_shot)
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
            #save_image(input.view(-1, 320, 320),f"{self.type}/{self.get_trajectory().shape[0]}", f"Images__{self.iter}")
            #self.iter += 1
            output = self.reconstruction_model(input)
            print(output.shape)
            output = transforms.root_sum_of_squares(transforms.complex_abs(input), dim=1).unsqueeze(1)
            return output.reshape(-1, 320, 320)

    def get_trajectory(self):
        return self.subsampling.get_trajectory()