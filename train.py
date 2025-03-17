import logging
import pathlib
import random
import shutil
import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import sys
# sys.path.insert(0, '/home/tomerweiss/multiPILOT2')

import numpy as np
# np.seterr('raise')
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from common.args import Args
from data import transforms
from data.mri_data import SliceData
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.subsampling_model import Subsampling_Model
from scipy.spatial import distance_matrix
from tsp_solver.greedy import solve_tsp
import scipy.io as sio
from common.utils import get_vel_acc
from common.evaluate import psnr, ssim
from fastmri.losses import SSIMLoss
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


# Example usage
folder_path = "output_images"
image_name = "example_image.png"


# save_image_without_pillow(tensor_image, folder_path, image_name)

class DataTransformCoils:
    def __init__(self, resolution):
        self.resolution = resolution

    def __call__(self, kspace, target, attrs, fname, slice):
        kspace = transforms.to_tensor(kspace)
        image = transforms.ifft2_regular(kspace)
        image = transforms.complex_center_crop(image, (self.resolution, self.resolution))
        image, mean, std = transforms.normalize_instance(image, eps=1e-11)
        target = transforms.to_tensor(target)
        target, mean, std = transforms.normalize_instance(target, eps=1e-11)
        mean = std = 0



        if target.shape[1] != self.resolution:
            target = transforms.center_crop(target, (self.resolution, self.resolution))
        return image , target, mean, std, attrs['norm'].astype(np.float32)

class DataTransform:
    def __init__(self, resolution):
        self.resolution = resolution

    def __call__(self, kspace, target, attrs, fname, slice):
        kspace = transforms.to_tensor(kspace)
        image = transforms.ifft2_regular(kspace)
        image = transforms.complex_center_crop(image, (self.resolution, self.resolution))
        image, mean, std = transforms.normalize_instance(image, eps=1e-11)
        target = transforms.to_tensor(target)
        target, mean, std = transforms.normalize_instance(target, eps=1e-11)
        mean = std = 0



        if target.shape[1] != self.resolution:
            target = transforms.center_crop(target, (self.resolution, self.resolution))
        return image.mean(0) , target, mean, std, attrs['norm'].astype(np.float32)


def create_datasets(args):
    print(args.data_path / f'multicoil_train', flush=True)
    print(args.data_path / f'multicoil_val', flush=True)
    train_data = SliceData(
        root=args.data_path / f'multicoil_train',
        transform=DataTransform(args.resolution),
        sample_rate=args.sample_rate)

    dev_data = SliceData(
        root=args.data_path / f'multicoil_val',
        transform=DataTransform(args.resolution),
        sample_rate=args.sample_rate)

    return dev_data, train_data


def create_data_loaders(args):
    dev_data, train_data = create_datasets(args)
    display_data = [dev_data[i] for i in range(0, len(dev_data), 1 if len(dev_data) // 16 == 0 else len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=20,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=20,
        pin_memory=True,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=args.batch_size,
        num_workers=20,
        pin_memory=True,
    )
    return train_loader, dev_loader, display_loader


def tsp_solver(x):
    d = distance_matrix(x, x)
    t = solve_tsp(d)
    return x[t, :]

def optimize_trajectory(args, model):
    if not args.trajectory_learning:
        return

    optimizer = torch.optim.Adam(model.module.parameters(), lr=1e-1)
    a_max, v_max = args.a_max, args.v_max
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100,
                                                           verbose=False)

    x = model.module.get_trajectory()
    v, a = get_vel_acc(x)
    init_loss = torch.sqrt(torch.sum(torch.pow(F.softshrink(a, a_max).abs() + 1e-8, 2))) + torch.sqrt(torch.sum(torch.pow(F.softshrink(v, v_max).abs() + 1e-8, 2)))
    iteration = 0
    while iteration < 1000:
        optimizer.zero_grad()
        x = model.module.get_trajectory()
        v, a = get_vel_acc(x)
        acc_loss = torch.sqrt(torch.sum(torch.pow(F.softshrink(a, a_max).abs() + 1e-8, 2)))
        vel_loss = torch.sqrt(torch.sum(torch.pow(F.softshrink(v, v_max).abs() + 1e-8, 2)))
        loss = acc_loss + vel_loss
        loss.backward()
        optimizer.step()

        if acc_loss.item() <  1e-5 and vel_loss.item() < 1e-5:
            print(f"acc_loss reached {acc_loss.item()} at iteration {iteration + 1}. Stopping training.")
            return
        scheduler.step(loss)
        iteration += 1

    print("Training completed!, took it from: ", init_loss," to: ", acc_loss + vel_loss)

def train_epoch(args, epoch, model, data_loader, optimizer, writer, scheduler):
    model.train()
    avg_loss = 0.

    if args.inter_gap_mode == "changing_downwards":
        if epoch < 10:
            model.module.subsampling.interp_gap = 50
        elif epoch == 10:
            model.module.subsampling.interp_gap = 30
        elif epoch == 15:
            model.module.subsampling.interp_gap = 20
        elif epoch == 20:
            model.module.subsampling.interp_gap = 10
        elif epoch == 23:
            model.module.subsampling.interp_gap = 5
        elif epoch == 25:
            model.module.subsampling.interp_gap = 1

    if args.inter_gap_mode == "changing_upwards":
        if epoch < 10:
            model.module.subsampling.interp_gap = 1
        elif epoch == 10:
            model.module.subsampling.interp_gap = 5
        elif epoch == 15:
            model.module.subsampling.interp_gap = 10
        elif epoch == 20:
            model.module.subsampling.interp_gap = 20
        elif epoch == 23:
            model.module.subsampling.interp_gap = 30
        elif epoch == 25:
            model.module.subsampling.interp_gap = 50

    print("\n", "epochs: ", epoch ," model.module.subsampling.interp_gap: ", model.module.subsampling.interp_gap)

    psnr_l = []
    ssim_l = []
    start_epoch = start_iter = time.perf_counter()
    print(f'a_max={args.a_max}, v_max={args.v_max}')

    for iter, data in enumerate(data_loader):
        torch.cuda.empty_cache()
        optimizer.zero_grad()

        input, target, mean, std, norm = data
        input = input.to(args.device)
        target = target.to(args.device)

        output = model(input.unsqueeze(1))

        x = model.module.get_trajectory()
        v, a = get_vel_acc(x)
        acc_loss = torch.sqrt(torch.sum(torch.pow(F.softshrink(a, args.a_max).abs() + 1e-8, 2)))
        vel_loss = torch.sqrt(torch.sum(torch.pow(F.softshrink(v, args.v_max).abs() + 1e-8, 2)))
        resolution = target.shape[-1]

        if epoch <= 10:
            vel_weight = 1e-3
        elif epoch <= 20:
            vel_weight = 1e-2
        elif epoch <= 30:
            vel_weight = 1e-1

        if epoch <= 10:
            acc_weight = 1e-3
        elif epoch <= 20:
            acc_weight = 1e-2
        elif epoch <= 30:
            acc_weight = 1e-1

        if iter % 100:
            save_dir = f"Images_{args.model}/{args.test_name}/{epoch}"
            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}/{iter}.png"
            save_image(torch.stack([output[0].view(resolution, resolution),
                                    target[0].view(resolution, resolution)]), save_path, f"{iter}")

        data_min = target.min()
        data_max = target.max()
        target_normalized = (target - data_min) / (data_max - data_min)
        output_normalized = (output - data_min) / (data_max - data_min)

        loss_l1 = F.l1_loss(output, target)
        psnr_l.append(psnr(target.detach().cpu().numpy(), output.detach().cpu().numpy()))
        ssim_l.append(ssim(target.detach().cpu().numpy(), output.detach().cpu().numpy()))
        rec_loss = loss_l1 #+ dcLoss # SSIMLoss().to(args.device)(output, target, data_range) # F.l1_loss(output, target)
        if args.TSP and epoch < args.TSP_epoch:
            loss = args.rec_weight * rec_loss
        else:
            loss = args.rec_weight * rec_loss + args.vel_weight * vel_loss + args.acc_weight * acc_loss

        #if vel_loss + acc_loss > 1e-3:
        #    optimize_trajectory(args, model)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=1.)
        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        # writer.add_scalar('TrainLoss', loss.item(), global_step + iter)

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'rec_loss: {rec_loss:.4g}, vel_loss: {vel_loss:.4g}, acc_loss: {acc_loss:.4g}'
                f'PSNR: {np.mean(psnr_l):.2f} +- {np.std(psnr_l):.2f}, SSIM: {np.mean(ssim_l):.4f} +- {np.std(ssim_l):.4f}'
            )
        start_iter = time.perf_counter()
    if scheduler:
        scheduler.step()
    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer):
    model.eval()
    losses = []
    psnr_l = []
    ssim_l = []

    start = time.perf_counter()
    with torch.no_grad():
        if epoch > 0:
            for iter, data in enumerate(data_loader):
                input, target, mean, std, norm = data
                input = input.to(args.device)
                resolution = target.shape[-1]
                target = target.to(args.device)
                target = transforms.reflect_pad_to_shape(target.unsqueeze(0), (resolution, resolution))

                output = model(input.unsqueeze(1))
                recons = output.to('cpu').squeeze(1).view(target.shape)
                recons = recons.squeeze()
                if output.shape != target.shape:
                    target = target.view_as(output)
                loss = F.l1_loss(output, target)
                losses.append(loss.item())
                target = target.view(-1,resolution,resolution)
                recons = recons.view(target.shape)

                psnr_l.append(psnr(target.to('cpu').numpy(), recons.numpy()))
                ssim_l.append(ssim(target.to('cpu').numpy(), recons.numpy()))

        print(
            f'PSNR: {np.mean(psnr_l):.2f} +- {np.std(psnr_l):.2f}, SSIM: {np.mean(ssim_l):.4f} +- {np.std(ssim_l):.4f}')
        x = model.module.get_trajectory()
        v, a = get_vel_acc(x)
        acc_loss = torch.sqrt(torch.sum(torch.pow(F.softshrink(a, args.a_max), 2)))
        vel_loss = torch.sqrt(torch.sum(torch.pow(F.softshrink(v, args.v_max), 2)))
        rec_loss = np.mean(losses)

        writer.add_scalar('Rec_Loss', rec_loss, epoch)
        writer.add_scalar('Acc_Loss', acc_loss.detach().cpu().numpy(), epoch)
        writer.add_scalar('Vel_Loss', vel_loss.detach().cpu().numpy(), epoch)
        writer.add_scalar('Total_Loss',
                          rec_loss + acc_loss.detach().cpu().numpy() + vel_loss.detach().cpu().numpy(), epoch)

        psnr_mean, psnr_std = np.mean(psnr_l), np.std(psnr_l)
        ssim_mean, ssim_std = np.mean(ssim_l), np.std(ssim_l)
        x = model.module.get_trajectory()
        v, a = get_vel_acc(x)
        if args.TSP and epoch < args.TSP_epoch:
            writer.add_figure('Scatter', plot_scatter(x.detach().cpu().numpy()), epoch)
        else:
            trajectory = plot_trajectory(x.detach().cpu().numpy())
            ax = trajectory.gca()  # Get the current axis from the figure
            text_str = f'PSNR: {psnr_mean:.2f} ± {psnr_std:.2f}\nSSIM: {ssim_mean:.4f} ± {ssim_std:.4f}'
            ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            save_dir = f"trajectory/{args.exp_dir}_{args.model}/{args.interp_gap}"
            os.makedirs(save_dir, exist_ok=True)
            trajectory.savefig(f"{save_dir}/trajectory_epoch_{epoch}.png")
            plt.close(trajectory)
            writer.add_figure('Trajectory', plot_trajectory(x.detach().cpu().numpy()), epoch)
            writer.add_figure('Scatter', plot_scatter(x.detach().cpu().numpy()), epoch)
        writer.add_figure('Accelerations_plot', plot_acc(a.cpu().numpy(), args.a_max), epoch)
        writer.add_figure('Velocity_plot', plot_acc(v.cpu().numpy(), args.v_max), epoch)
        writer.add_text('Coordinates', str(x.detach().cpu().numpy()).replace(' ', ','), epoch)
    if epoch == 0:
        return None, time.perf_counter() - start
    else:
        return np.mean(losses), time.perf_counter() - start, psnr_mean, ssim_mean


def plot_scatter(x):
    fig = plt.figure(figsize=[10, 10])
    ax = fig.add_subplot(1, 1, 1)
    ax.axis([-165, 165, -165, 165])
    for i in range(x.shape[0]):
        ax.plot(x[i, :, 0], x[i, :, 1], '.')
    return fig


def plot_trajectory(x):
    fig = plt.figure(figsize=[10, 10])
    ax = fig.add_subplot(1, 1, 1)
    ax.axis([-165, 165, -165, 165])
    for i in range(x.shape[0]):
        ax.plot(x[i, :, 0], x[i, :, 1])
    return fig


def plot_acc(a, a_max=None):
    fig, ax = plt.subplots(2, sharex=True)
    for i in range(a.shape[0]):
        ax[0].plot(a[i, :, 0])
        ax[1].plot(a[i, :, 1])
    if a_max is not None:
        limit = np.ones(a.shape[1]) * a_max
        ax[1].plot(limit, color='red')
        ax[1].plot(-limit, color='red')
        ax[0].plot(limit, color='red')
        ax[0].plot(-limit, color='red')
    return fig


def visualize(args, epoch, model, data_loader, writer):
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        print("entered visualize: ")
        for iter, data in enumerate(data_loader):
            input, target, mean, std, norm = data
            input = input.to(args.device)
            target = target.unsqueeze(1).to(args.device)

            save_image(target, 'Target')
            if epoch != 0:
                print(input.unsqueeze(1).shape)
                output = model(input.unsqueeze(1))
                save_image(output, 'Reconstruction')
                save_image(torch.abs(target - output), 'Error')
            break


def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, best_psnr_mean, best_ssim_mean, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'best_psnr_mean': best_psnr_mean,
            'best_ssim_mean': best_ssim_mean,
            'exp_dir': exp_dir
        },
        f=exp_dir + '/model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir + '/model.pt', exp_dir + '/best_model.pt')


def build_model(args):
    if args.end_epsilon != None and args.epsilon != None:
        epsilon_step = (args.end_epsilon - args.epsilon)/args.num_epochs
    else:
        epsilon_step = 0

    model = Subsampling_Model(
        in_chans=args.in_chans,
        out_chans=args.out_chans,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob,
        decimation_rate=args.decimation_rate,
        res=args.resolution,
        trajectory_learning=args.trajectory_learning,
        initialization=args.initialization,
        SNR=args.SNR,
        n_shots=args.n_shots,
        interp_gap=args.interp_gap,
        type=args.model,
        img_size=args.img_size,
        window_size=args.window_size,
        embed_dim=args.embed_dim,
        num_blocks=args.num_blocks,
        sample_per_shot=args.sample_per_shot,
        noise_mode = args.noise_mode,
        epsilon = args.epsilon,
        epsilon_step = epsilon_step,
        noise_type = args.noise_type,
        noise_p = args.noise_p
    ).to(args.device)
    return model


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer, args


def build_optim(args, model):
    optimizer = torch.optim.Adam([{'params': model.module.subsampling.parameters(), 'lr': args.sub_lr},
                                  {'params': model.module.reconstruction_model.parameters()}], args.lr)
    return optimizer


def eval(args, model, data_loader):
    model.eval()
    psnr_l = []
    ssim_l = []
    with torch.no_grad():
        for (input, target, mean, std, norm) in data_loader:
            input = input.to(args.device)
            recons = model(input.unsqueeze(1)).to('cpu').squeeze(1)
            # recons = transforms.complex_abs(recons)  # complex to real
            recons = recons.squeeze()
            target = target.to('cpu')

            psnr_l.append(psnr(target.numpy(), recons.numpy()))
            ssim_l.append(ssim(target.numpy(), recons.numpy()))

    print(f'PSNR: {np.mean(psnr_l):.2f} +- {np.std(psnr_l):.2f}, SSIM: {np.mean(ssim_l):.4f} +- {np.std(ssim_l):.4f}')
    return

def train():
    import torch
    if torch.cuda.is_available():
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")  # 0 is the index of the first GPU
    else:
        print("CUDA device not available")
    torch.cuda.empty_cache()
    print("started training", flush=True)
    args = create_arg_parser().parse_args()

    args.v_max = args.gamma * args.G_max * args.FOV * args.dt
    args.exp_dir = f'summary/{args.test_name}'

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    pathlib.Path(args.exp_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.exp_dir)
    with open(args.exp_dir + '/args.txt', "w") as text_file:
        print(vars(args), file=text_file)

    print(args.test_name, flush=True)
    args.checkpoint = f'summary/{args.test_name}/model.pt'
    if args.resume:
        checkpoint, model, optimizer = load_model(args.checkpoint)
        # args = checkpoint['args']
        best_dev_loss = checkpoint['best_dev_loss']
        best_psnr_mean = checkpoint['best_psnr_mean']
        best_ssim_mean = checkpoint['best_ssim_mean']
        start_epoch = checkpoint['epoch'] + 1
        del checkpoint
    else:
        model = build_model(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = build_optim(args, model)
        best_dev_loss = 1e9
        best_psnr_mean = 0
        best_ssim_mean = 0
        start_epoch = 0
    inter_gap_mode = args.inter_gap_mode
    noise_behaviour = args.noise_behaviour
    logging.info(args)

    train_loader, dev_loader, display_loader = create_data_loaders(args)
    dev_loss, dev_time = evaluate(args, 0, model, dev_loader, writer)
    print("started mid point", flush=True)
    for epoch in range(start_epoch, args.num_epochs):
        if noise_behaviour == "log":
            model.module.subsampling.epsilon = np.logspace(np.log10(args.epsilon), np.log10(args.end_epsilon), num=args.num_epochs)[epoch]
        print("epoch: ", epoch, "current noise level: ", model.module.subsampling.epsilon, "started", flush=True)
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, writer, None)
        dev_loss, dev_time, psnr_mean, ssim_mean = evaluate(args, epoch + 1, model, dev_loader, writer)

        if psnr_mean > best_psnr_mean:
            is_new_best = True
            best_psnr_mean = psnr_mean
            best_ssim_mean = ssim_mean
            best_epoch = epoch + 1
        else:
            is_new_best = False
        save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, best_psnr_mean, best_ssim_mean, is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'DevLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )

        if noise_behaviour == "linear":
            model.module.increase_noise_linearly()

    print(args.test_name)
    eval(args, model, dev_loader)
    print(f'Training done, best epoch: {best_epoch}')
    writer.close()


def create_arg_parser():
    parser = Args()
    parser.add_argument('--test-name', type=str, default='gaussiantsp-d24-a1e-3-v1e-3', help='name for the output dir')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='summary/testepi',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str, default='summary/test/model.pt',
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--report-interval', type=int, default=1, help='Period of loss reporting')

    # model parameters
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')
    parser.add_argument('--data-parallel', action='store_true', default=False,
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--decimation-rate', default=10, type=int,
                        help='Ratio of k-space columns to be sampled. If multiple values are '
                             'provided, then one of those is chosen uniformly at random for each volume.')

    # optimization parameters
    parser.add_argument('--batch-size', default=9, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=30,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.01,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    parser.add_argument('--sub-lr', type=float, default=1e-1, help='lerning rate of the sub-samping layel')

    # trajectory learning parameters
    parser.add_argument('--trajectory-learning', default=True,
                        help='trajectory_learning, if set to False, fixed trajectory, only reconstruction learning.')
    parser.add_argument('--acc-weight', type=float, default=1e-2, help='weight of the acceleration loss')
    parser.add_argument('--vel-weight', type=float, default=1e-1, help='weight of the velocity loss')
    parser.add_argument('--rec-weight', type=float, default=1, help='weight of the reconstruction loss')
    parser.add_argument('--gamma', type=float, default=42576, help='gyro magnetic ratio - kHz/T')
    parser.add_argument('--G-max', type=float, default=40, help='maximum gradient (peak current) - mT/m')
    parser.add_argument('--S-max', type=float, default=200, help='maximum slew-rate - T/m/s')
    parser.add_argument('--FOV', type=float, default=0.2, help='Field Of View - in m')
    parser.add_argument('--dt', type=float, default=1e-5, help='sampling time - sec')
    parser.add_argument('--a-max', type=float, default=0.17, help='maximum acceleration')
    parser.add_argument('--v-max', type=float, default=3.4, help='maximum velocity')
    parser.add_argument('--TSP', action='store_true', default=False,
                        help='Using the PILOT-TSP algorithm,if False using PILOT.')
    parser.add_argument('--TSP-epoch', default=20, type=int, help='Epoch to preform the TSP reorder at')
    parser.add_argument('--initialization', type=str, default='radial',
                        help='Trajectory initialization when using PILOT (spiral, EPI, rosette, uniform, gaussian).')
    parser.add_argument('--SNR', action='store_true', default=False,
                        help='add SNR decay')
    parser.add_argument('--n-shots', type=int, default=16,
                        help='Number of shots')
    parser.add_argument('--interp_gap', type=int, default=10,
                        help='number of interpolated points between 2 parameter points in the trajectory')
    parser.add_argument('--model', type=str, default='Unet',
                        help='the model type and params of the reconstruction net')
    parser.add_argument('--inter-gap-mode', type=str, default='constant',
                        help='How the interpolated gap will change during the training')
    parser.add_argument('--img-size', type=int, nargs=2, default=[320, 320], help='Image size (height, width)')
    parser.add_argument('--in-chans', type=int, default=1, help='Number of input channels')
    parser.add_argument('--out-chans', type=int, default=1, help='Number of output channels')
    parser.add_argument('--num-blocks', type=int, default=1, help='Number of blocks in the model')
    parser.add_argument('--window-size', type=int, default=10, help='Window size for the model')
    parser.add_argument('--embed-dim', type=int, default=66, help='Embedding dimension for the model')
    parser.add_argument('--sample-per-shot', type=int, default=3001, help='Number of samples per shot')
    parser.add_argument('--noise-mode', type=str, default='ones',
                        help='Type of noise to be added (e.g., "ones" or "random")')
    parser.add_argument('--noise-behaviour', type=str, default='constant',
                        help='How the noise should behave (e.g., "constant" or "linear")')
    parser.add_argument('--epsilon', type=float, default=0, help='Starting value of epsilon for noise scaling')
    parser.add_argument('--end-epsilon', type=float, default=1e7, help='End value of epsilon for noise scaling')
    parser.add_argument('--noise-type', type=str, default='l1', help='Type of noise to be added (e.g., "l1", "l2")')
    parser.add_argument('--noise-p', type=float, default='0', help='Probability of applying noise during training')
    return parser


if __name__ == '__main__':
    train()
