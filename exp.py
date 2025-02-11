#!/home/mohammed-wa/miniconda3/envs/pilot/bin/python
import os
import json
import shlex

data_path = '/home/mohammed-wa/PycharmProjects/MPILOT/Datasets/pd_only'

sub_lr = 1e-5
rec_lr = 0.0001
acc_weight = 0.01
vel_weight = 0.01
batch_size = 1
init = 'spiral_high'
n_shots = 16 # TODO try to increase (maybe to 20, 24) to increase the PSNR
interp_gap = 1
trajectory_learning = 1
num_epochs = 100
sample_rate = 1.
TSP = ''
SNR = ''
weight_decay = 0
model_params = {
    "model": 'Unet',
    "img_size": [320, 320],
    "in_chans": 1,
    "out_chans": 1,
    "num_blocks": 1,
    "window_size": 10,
    "embed_dim": 33
}

model_params_str = json.dumps(model_params)
model_params_str_escaped = shlex.quote(model_params_str)

print(model_params_str)
if trajectory_learning == 1:
    test_name = f'{n_shots}/{init}_{sub_lr}_{acc_weight}_{vel_weight}'
else:
    test_name = f'{n_shots}/{init}_fixed'

if TSP == '--TSP':
    test_name = f'{n_shots}/{init}_TSP_{sub_lr}_{acc_weight}_{vel_weight}'

if SNR == '--SNR':
    test_name += '_SNR_flat_0.01'

test_name += f'_{model_params_str_escaped}'
# train
os.system(f'python3 train.py --test-name={test_name} --n-shots={n_shots}'
         f' --trajectory-learning={trajectory_learning} --sub-lr={sub_lr}  --initialization={init} '
         f'--batch-size={batch_size}  --lr={rec_lr} --num-epochs={num_epochs} --acc-weight={acc_weight} '
         f'--vel-weight={vel_weight} --data-path={data_path} --sample-rate={sample_rate}'
         f' --data-parallel {TSP} {SNR} --weight-decay={weight_decay} --model={model_params_str_escaped}')
