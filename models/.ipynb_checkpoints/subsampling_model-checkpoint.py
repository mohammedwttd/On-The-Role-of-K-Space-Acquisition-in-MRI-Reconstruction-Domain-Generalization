import torch
from torch import nn
from torch.nn import functional as F

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

import torch
import numpy as np




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
        sampel_per_shot = 3001
        if initialization == 'spiral':
            x = np.load(f'spiral/{n_shots}int_spiral_low.npy')
            x = torch.tensor(x[:, :sampel_per_shot, :]).float()
        elif initialization == 'spiral_high':
            x = np.load(f'spiral/{n_shots}int_spiral.npy') * 10
            x = torch.tensor(x[:, :sampel_per_shot, :]).float()
        elif initialization == 'EPI':
            x = torch.zeros(n_shots, sampel_per_shot, 2)
            v_space = self.res // n_shots
            for i in range(n_shots):
                index = 0
                for j in range(sampel_per_shot):
                    x[i, index, 1] = (i + 0.5) * v_space - 160
                    x[i, index, 0] = j * 320 / sampel_per_shot - 160
                    index += 1
        elif initialization == 'radial':
            x = torch.zeros(n_shots, sampel_per_shot, 2)
            theta = np.pi / n_shots
            for i in range(n_shots):
                L = torch.arange(-160, 160, 320 / sampel_per_shot).float()
                x[i, :, 0] = L * np.cos(theta * i)
                x[i, :, 1] = L * np.sin(theta * i)
        elif initialization == 'uniform':
            x = (torch.rand(n_shots, sampel_per_shot, 2) - 0.5) * self.res
        elif initialization == 'gaussian':
            x = torch.randn(n_shots, sampel_per_shot, 2) * self.res / 6
        else:
            print('Wrong initialization')
        self.x = torch.nn.Parameter(x, requires_grad=bool(int(trajectory_learning)))
        return

    # check how projection_iters effects needed learn time
    def __init__(self, decimation_rate, res, trajectory_learning, initialization, n_shots, interp_gap, SNR=False,
                 projection_iters=10):
        super().__init__()

        self.decimation_rate = decimation_rate
        self.res = res
        self.num_measurements = res ** 2 // decimation_rate
        self.initilaize_trajectory(trajectory_learning, initialization, n_shots)
        self.SNR = SNR
        self.interp_gap = interp_gap
        self.iters = projection_iters

    def forward(self, input):
        # interpolate
        # TODO: fix the interpolation functionality.
        # if self.interp_gap > 1:
        t = torch.arange(0, self.x.shape[1], device=self.x.device).float()
        t1 = t[::self.interp_gap]
        x_short = self.x[:, ::self.interp_gap, :]

        # self.x.data = proj_handler(self.x.data, num_iters=self.iters)

        x_full = self.x.reshape(-1,3001,2)
        # TODO: understand why the permute is needed
        input = input.permute(0, 1, 4, 2, 3)
        sub_ksp = nufft(input, x_full)
        if self.SNR:
            noise_amp = 0.001
            noise = noise_amp * torch.randn(sub_ksp.shape)
            sub_ksp = sub_ksp + noise.to(sub_ksp.device)
        #print(sub_ksp.shape)
        #print(sub_ksp)
        output = nufft_adjoint(sub_ksp, x_full, input.shape)
        output = output.permute(0, 1, 3, 4, 2)
        #print(output.shape, input.shape)
        #mask = transforms.fft2(output) - transforms.fft2(input.permute(0, 1, 3, 4, 2))
        #print(mask.min(), mask.max())
        #print(transforms.fft2(output) - transforms.fft2(input.permute(0, 1, 3, 4, 2)))
        #mask = (transforms.fft2(output) - transforms.fft2(input.permute(0, 1, 3, 4, 2)) < 1).float()
        #print(mask.size)
        return output#, mask

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
    def __init__(self, img_size, in_chans, out_chans, num_blocks, window_size):
        super(HUMUSReconstructionModel, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                HUMUSBlock(
                    img_size=img_size,
                    in_chans=in_chans if i == 0 else out_chans,
                    out_chans=out_chans,
                    window_size=window_size
                )
            )
        self.final_block = HUMUSBlock(
            img_size=img_size,
            in_chans=out_chans,
            out_chans=1,
            window_size=window_size
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.final_block(x)
        return x


class HUMUSVarNet(nn.Module):
    """
    VarNet with HUMUSBlock for all cascades, where the last cascade has `out_chans=1`.
    """

    def __init__(self, img_size, in_chans, out_chans, window_size):
        """
        Args:
            img_size: Size of the input image for HUMUSBlock.
            in_chans: Number of input channels for HUMUSBlock.
            out_chans: Number of output channels for the last HUMUSBlock cascade.
            window_size: Window size parameter for HUMUSBlock.
        """
        super().__init__()

        # Define the VarNet with three cascades
        self.reconstruction_model = VarNet(
            num_cascades =5,
            reg_model_fn = lambda cascade_index: (
                HUMUSBlock(img_size=img_size, in_chans=in_chans, out_chans=out_chans, window_size=window_size)
            ),
        )
        #self.tail = HUMUSBlock(img_size=img_size, in_chans=in_chans, out_chans=1, window_size=window_size)

    def forward(self, kspace, ref_kspace):
        """
        Args:
            kspace: Input k-space data of shape [batch_size, num_coils, height, width].
            ref_kspace: Reference k-space data of the same shape.

        Returns:
            Refined k-space data.
        """
        image = transforms.ifft2_regular(self.reconstruction_model(kspace, ref_kspace)).view(-1,1,320,320,2)#.permute(0, 3, 1, 2)
        #print(mid.shape)
        #print(transforms.complex_abs(image).shape)
        return transforms.root_sum_of_squares(transforms.complex_abs(image), dim=1)




class Subsampling_Model(nn.Module):
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob, decimation_rate, res,
                 trajectory_learning, initialization, n_shots, interp_gap, SNR=False, type ='vit'):
        super().__init__()
        self.subsampling = Subsampling_Layer(decimation_rate, res, trajectory_learning, initialization, n_shots,
                                             interp_gap, SNR)
        self.type = type
        avrg_img_size = 320
        patch_size = 10
        depth = 4
        num_heads = 9
        embed_dim = 300
        from fastmri.data.subsample import create_mask_for_mask_type

        print(self.type)
        if type == 'vit':
            vitNet = VisionTransformer(
                avrg_img_size=avrg_img_size,
                patch_size=patch_size,
                in_chans=1, embed_dim=embed_dim,
                depth=depth, num_heads=num_heads,
            )
             # ReconNet(net)
        elif type == 'Unet':
            print("got here")
            self.reconstruction_model = fastmri.models.Unet(in_chans, out_chans, chans, num_pool_layers, drop_prob)
        else:
            humusNet = HUMUSNet(use_checkpoint=False, num_cascades=4, img_size=[320, 320], window_size=4,
                                mask_center=False,num_adj_slices=1)
            self.reconstruction_model = humusNet
            self.mask = nn.Parameter(torch.ones((320, 320), dtype=torch.float32))

    def subsample_and_stack(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies subsampling to each coil's k-space measurements and stacks the results.

        Args:
            input_tensor (torch.Tensor): A tensor of shape (num_coils, ...) where each coil's
                                         measurements are independently subsampled.

        Returns:
            torch.Tensor: A tensor of shape (num_coils, 1, ...) with each coil's subsampled result.
        """
        # Apply subsampling for each coil and stack along the first dimension
        print(input_tensor.shape)
        subsampled_results = [self.subsampling(input_tensor[i]).unsqueeze(0) for i in range(input_tensor.shape[0])]

        # Stack the results along the first dimension
        stacked_results = torch.cat(subsampled_results, dim=0)

        return stacked_results.unsqueeze(1)

    def forward(self, input):
        if self.type == "vit":
            input = self.subsample_and_stack(input).squeeze()
            kspace = transforms.fft2(input)
            mask = self.subsampling.trajectory_to_mask((kspace.shape[-3], kspace.shape[-2]))
            print(mask.shape)

            # Count the number of ones
            num_ones = torch.sum(mask == 1).item()
            input = transforms.root_sum_of_squares(transforms.complex_abs(input), dim=0).unsqueeze(0).unsqueeze(0)
            return self.reconstruction_model(input, None)
        elif self.type == 'Unet':
            input = self.subsampling(input.squeeze(0)).reshape(-1,1,320,320,2)
            alsoReturn = transforms.complex_abs(input).reshape(-1,320,320)
            input = transforms.root_sum_of_squares(transforms.complex_abs(input), dim=0)
            output = self.reconstruction_model(input.unsqueeze(0)).reshape(-1, 320, 320)
            return output, alsoReturn
        elif self.type == "humus":
            input = input.squeeze(0)
            print(input.shape)
            input = self.subsampling(input).squeeze()
            kspace = transforms.fft2(input).unsqueeze(0).reshape(1,-1,320,320,2)
            mask = torch.sigmoid(self.mask) > 0.5
            out =  self.reconstruction_model(kspace.to(input.device), mask.to(input.device).bool())
            return out
    def get_trajectory(self):
        return self.subsampling.get_trajectory()
