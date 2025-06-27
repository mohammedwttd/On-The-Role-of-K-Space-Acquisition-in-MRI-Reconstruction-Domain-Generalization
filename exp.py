#!/home/mohammed-wa/miniconda3/envs/mpilot/bin/python
import os
import json
import shlex
import sys

from sympy.physics.units import acceleration

data_path = '/mnt/walkure_public/users/mohammedw/fastmri_downloads/'

model = 'Unet'
init = 'radial'
sub_lr = 0.1 if 'cartesian' in init else 0.01
rec_lr = 1e-4 * 5 if "pretrained" not in model else 0.0001
acc_weight = 0.005
vel_weight = 0.001
batch_size = 1
n_shots = 16
trajectory_learning = 1 if "pretrained" in model else 1
num_epochs = 30 if "pretrained" in model else 40
sample_rate = 1 if "pretrained" in model else 1
TSP = ''
SNR = ''
weight_decay = 0
inter_gap_mode = "changing_downwards"
interp_gap = 10
acceleration = 4
center_fraction = 0.08

#model settings
img_size = [320, 320]
in_chans = 1
out_chans = 1
num_blocks = 1
sample_per_shot = 1600
drop_prob = 0.1

#relevant only for humus
window_size = 10
embed_dim = 66


#noise
noise_mode = None
noise_behaviour = "constant"
std = 0
std_image = 0
epsilon = 0
end_epsilon = 1e8
noise_type = "linf"
noise_steps = 10
noise_p = 0.5
test_name = f'{n_shots}/{init}_'

if init == "cartesian":
    test_name += f'{acceleration}_{center_fraction}_'
else:
    test_name += f'{n_shots}_{sample_per_shot}_'

if trajectory_learning == 1:
    test_name += f'{rec_lr}_{sub_lr}_{acc_weight}_{vel_weight}_{inter_gap_mode}_{interp_gap}_{model}_{num_epochs}'
else:
    test_name += f'{rec_lr}_fixed_{model}_{num_epochs}'

if TSP == '--TSP':
    test_name += f'{rec_lr}_TSP_{sub_lr}_{acc_weight}_{vel_weight}_{inter_gap_mode}'

if SNR == '--SNR':
    test_name += '_SNR_flat_0.01'

if epsilon != 0 and noise_behaviour == "PGD":
    test_name += f"_{noise_mode}"
    if noise_behaviour == "constant":
        test_name += "_constant_noise"

    if noise_behaviour == "linear":
        test_name += "_linear_noise"

    if noise_behaviour == "log":
        test_name += "_log_noise"

    test_name += "_PGD_noise"
    test_name += f"_steps{noise_steps}"

    test_name += f'_start_epsilon{epsilon}'

    if noise_behaviour == "linear":
        test_name += f'_end_epsilon{end_epsilon}'

    test_name += f'_noise_type{noise_type}'

    if noise_p == -1:
        test_name += f'_probabilistic_model'
    else:
        test_name += f'_P{noise_p}'

if std != 0:
    test_name += f"_{noise_behaviour}"
    test_name += f"_std_{std}"
    test_name += f"_std_image{std_image}"
    test_name += f"_noise_p_{noise_p}"

command = f'python3 train.py --test-name={test_name} ' \
          f'--n-shots={n_shots} ' \
          f'--trajectory-learning={trajectory_learning} ' \
          f'--sub-lr={sub_lr} ' \
          f'--initialization={init} ' \
          f'--batch-size={batch_size} ' \
          f'--lr={rec_lr} ' \
          f'--num-epochs={num_epochs} ' \
          f'--acc-weight={acc_weight} ' \
          f'--vel-weight={vel_weight} ' \
          f'--data-path={data_path} ' \
          f'--sample-rate={sample_rate} ' \
          f'--data-parallel {TSP} ' \
          f'--weight-decay={weight_decay} ' \
          f'--inter-gap-mode={inter_gap_mode} ' \
          f'--model={model} ' \
          f'--in-chans={in_chans} ' \
          f'--out-chans={out_chans} ' \
          f'--num-blocks={num_blocks} ' \
          f'--window-size={window_size} ' \
          f'--embed-dim={embed_dim} ' \
          f'--interp_gap={interp_gap} ' \
          f'--drop-prob={drop_prob} '\
          f'--sample-per-shot={sample_per_shot} ' \
          f'--noise-mode={noise_mode} ' \
          f'--noise-behaviour={noise_behaviour} ' \
          f'--epsilon={epsilon} ' \
          f'--end-epsilon={end_epsilon} ' \
          f'--noise-type={noise_type} '  \
          f'--noise-p={noise_p} '  \
          f'--noise-steps={noise_steps} '  \
          f'--std={std} '  \
          f'--std-image={std_image} '  \
          f'--acceleration={acceleration} '  \
          f'--center-fraction={center_fraction}'

os.system(command)
