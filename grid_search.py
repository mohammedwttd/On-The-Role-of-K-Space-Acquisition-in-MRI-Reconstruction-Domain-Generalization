import os
import json
import shlex
import itertools

sub_lr_values = [0.1, 0.15]
rec_lr_values = [0.001, 0.0005, 0.0001]
interp_gap_values = [1, 5, 10]
data_path = '/home/mohammed-wa/PycharmProjects/MPILOT/Datasets/pd_only'
acc_weight = 0
vel_weight = 0
batch_size = 1
init = 'radial'
n_shots = 16
trajectory_learning = 1
num_epochs = 40
sample_rate = 1
TSP = ''
SNR = ''
weight_decay = 0
inter_gap_mode = "constant"

model_params = {
    "model": 'Unet',
    "img_size": [320, 320],
    "in_chans": 1,
    "out_chans": 1,
    "num_blocks": 1,
    "window_size": 10,
    "embed_dim": 66,
    "sample_per_shot": 3001
}

model_params_str = json.dumps(model_params)
model_params_str_escaped = shlex.quote(model_params_str)

os.makedirs("slurm_logs", exist_ok=True)

for sub_lr, rec_lr, interp_gap in itertools.product(sub_lr_values, rec_lr_values, interp_gap_values):
    test_name = f'{n_shots}/{init}_{rec_lr}_{sub_lr}_{acc_weight}_{vel_weight}_{inter_gap_mode}_{interp_gap}'

    if TSP == '--TSP':
        test_name = f'{n_shots}/{init}_{rec_lr}_TSP_{sub_lr}_{acc_weight}_{vel_weight}_{inter_gap_mode}_{interp_gap}'

    if SNR == '--SNR':
        test_name += '_SNR_flat_0.01'

    test_name += f'_{model_params_str_escaped}'

    command = f'./py-sbatch.sh train.py --test-name={n_shots}/{init}_{rec_lr}_{sub_lr}_{acc_weight}_{vel_weight}_{inter_gap_mode} ' \
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
              f'--interp_gap={interp_gap} ' \
              f'--inter-gap-mode={inter_gap_mode} ' \
              f'--model={model_params["model"]} ' \
              f'--in-chans={model_params["in_chans"]} ' \
              f'--out-chans={model_params["out_chans"]} ' \
              f'--num-blocks={model_params["num_blocks"]} ' \
              f'--window-size={model_params["window_size"]} ' \
              f'--embed-dim={model_params["embed_dim"]} ' \
              f'--sample-per-shot={model_params["sample_per_shot"]} '

    os.system(command)
    print(f"Submitted job: {test_name}")
