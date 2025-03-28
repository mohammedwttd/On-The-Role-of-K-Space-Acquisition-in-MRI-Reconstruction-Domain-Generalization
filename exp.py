#!/home/mohammed-wa/miniconda3/envs/mpilot/bin/python
import os
import json
import shlex

data_path = '/home/mohammed-wa/PycharmProjects/MPILOT/Datasets/pd_only'

sub_lr = 0.075
rec_lr = 1e-4 * 5
acc_weight = 0.001
vel_weight = 0.001
batch_size = 10
init = 'radial'
n_shots = 16
trajectory_learning = 1
num_epochs = 50
sample_rate = 1
TSP = ''
SNR = ''
weight_decay = 0
inter_gap_mode = "changing_downwards"
interp_gap = 10

#model settings
model = 'Unet'
img_size = [320, 320]
in_chans = 1
out_chans = 1
num_blocks = 1
sample_per_shot = 3001
drop_prob = 0.1

#relevant only for humus
window_size = 10
embed_dim = 66


#noise
noise_mode = "random"
noise_behaviour = "PGD"
epsilon = 3
end_epsilon = 1e8
noise_type = "linf"
noise_steps = 2
noise_p = 0.5             #-1 for having p * epsilon

if trajectory_learning == 1:
    test_name = f'{n_shots}/{init}_{rec_lr}_{sub_lr}_{acc_weight}_{vel_weight}_{inter_gap_mode}_{interp_gap}_{model}'
else:
    test_name = f'{n_shots}/{init}_{rec_lr}_fixed'

if TSP == '--TSP':
    test_name = f'{n_shots}/{init}_{rec_lr}_TSP_{sub_lr}_{acc_weight}_{vel_weight}_{inter_gap_mode}'

if SNR == '--SNR':
    test_name += '_SNR_flat_0.01'

if epsilon != 0:
    test_name += f"_{noise_mode}"
    if noise_behaviour == "constant":
        test_name += "_constant_noise"

    if noise_behaviour == "linear":
        test_name += "_linear_noise"

    if noise_behaviour == "log":
        test_name += "_log_noise"

    if noise_behaviour == "PGD":
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


# train
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
          f'--noise-steps={noise_steps}'

print(command)
os.system(command)