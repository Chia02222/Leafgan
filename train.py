import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import csv
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from skimage.metrics import structural_similarity as ssim

def calculate_mse(real_images, reconstructed_images):
    device = real_images.device
    real_images = real_images.to(device)
    reconstructed_images = reconstructed_images.to(device)
    mse = torch.nn.functional.mse_loss(real_images, reconstructed_images)
    return mse.item()

def calculate_psnr(real_images, reconstructed_images):
    device = real_images.device
    real_images = real_images.to(device)
    reconstructed_images = reconstructed_images.to(device)
    mse = torch.nn.functional.mse_loss(real_images, reconstructed_images)
    psnr = 10 * torch.log10(1 / mse)
    return psnr.item()

def save_metrics_plot(epoch_mse_A, epoch_psnr_A, epoch_mse_B, epoch_psnr_B, checkpoint_dir):
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(range(len(epoch_mse_A)), epoch_mse_A, label='Average MSE A')
    plt.xlabel('Epoch')
    plt.ylabel('Average MSE A')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(range(len(epoch_psnr_A)), epoch_psnr_A, label='Average PSNR A')
    plt.xlabel('Epoch')
    plt.ylabel('Average PSNR A')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(range(len(epoch_mse_B)), epoch_mse_B, label='Average MSE B')
    plt.xlabel('Epoch')
    plt.ylabel('Average MSE B')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(range(len(epoch_psnr_B)), epoch_psnr_B, label='Average PSNR B')
    plt.xlabel('Epoch')
    plt.ylabel('Average PSNR B')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'metrics_plot.png'))
    plt.close()

def save_metrics_csv(epoch_mse_A, epoch_psnr_A, epoch_mse_B, epoch_psnr_B, epoch_losses, checkpoint_dir):
    csv_file = os.path.join(checkpoint_dir, 'metrics.csv')
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Average MSE A', 'Average PSNR A', 'Average MSE B', 'Average PSNR B', 'Loss'])
        for epoch in range(len(epoch_mse_A)):
            writer.writerow([epoch + 1, epoch_mse_A[epoch], epoch_psnr_A[epoch], epoch_mse_B[epoch], epoch_psnr_B[epoch], epoch_losses[epoch]])

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = 0

    mse_list_A = []
    psnr_list_A = []
    mse_list_B = []
    psnr_list_B = []
    epoch_mse_A = []
    epoch_psnr_A = []
    epoch_mse_B = []
    epoch_psnr_B = []
    epoch_losses = []

    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            real_A = data['A'].to(model.device)
            real_B = data['B'].to(model.device)
            visuals = model.get_current_visuals()

            rec_A_key = 'rec_A'
            rec_B_key = 'rec_B'
            if rec_A_key in visuals:
                rec_A = visuals[rec_A_key].to(model.device)

                mse_A = calculate_mse(real_A, rec_A)
                psnr_A = calculate_psnr(real_A, rec_A)

                mse_list_A.append(mse_A)
                psnr_list_A.append(psnr_A)

                if total_iters % opt.print_freq == 0:
                    print(f'MSE A: {mse_A}')
                    print(f'PSNR A: {psnr_A}')
            else:
                print(f"Key '{rec_A_key}' not found in visuals")

            if rec_B_key in visuals:
                rec_B = visuals[rec_B_key].to(model.device)

                mse_B = calculate_mse(real_B, rec_B)
                psnr_B = calculate_psnr(real_B, rec_B)

                mse_list_B.append(mse_B)
                psnr_list_B.append(psnr_B)

                if total_iters % opt.print_freq == 0:
                    print(f'MSE B: {mse_B}')
                    print(f'PSNR B: {psnr_B}')
            else:
                print(f"Key '{rec_B_key}' not found in visuals")

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                print(f'Epoch: {epoch}, Iteration: {epoch_iter}, Losses: {losses}, Time per batch: {t_comp:.4f}s, Data loading time: {t_data:.4f}s')

            if total_iters % opt.save_latest_freq == 0:
                print(f'Saving the latest model (epoch {epoch}, total_iters {total_iters})')
                save_suffix = f'iter_{total_iters}' if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print(f'Saving the model at the end of epoch {epoch}, iters {total_iters}')
            model.save_networks('latest')
            model.save_networks(epoch)

            # Calculate and store average metrics for the epoch
            avg_mse_A = np.mean(mse_list_A)
            avg_psnr_A = np.mean(psnr_list_A)
            avg_mse_B = np.mean(mse_list_B)
            avg_psnr_B = np.mean(psnr_list_B)
            avg_loss = np.mean([losses[k] for k in losses])  # Calculate average loss
            epoch_mse_A.append(avg_mse_A)
            epoch_psnr_A.append(avg_psnr_A)
            epoch_mse_B.append(avg_mse_B)
            epoch_psnr_B.append(avg_psnr_B)
            epoch_losses.append(avg_loss)

            print(f'End of epoch {epoch} / {opt.niter + opt.niter_decay} \t Time Taken: {time.time() - epoch_start_time:.2f} sec')
            print(f'Epoch {epoch} - Average MSE A: {avg_mse_A}, Average PSNR A: {avg_psnr_A}')
            print(f'Epoch {epoch} - Average MSE B: {avg_mse_B}, Average PSNR B: {avg_psnr_B}')
            print(f'Epoch {epoch} - Average Loss: {avg_loss}')
            model.update_learning_rate()

            # Save metrics plot and CSV
            save_metrics_plot(epoch_mse_A, epoch_psnr_A, epoch_mse_B, epoch_psnr_B, checkpoint_dir)
            save_metrics_csv(epoch_mse_A, epoch_psnr_A, epoch_mse_B, epoch_psnr_B, epoch_losses, checkpoint_dir)

            # Clear metrics for the next epoch
            mse_list_A = []
            psnr_list_A = []
            mse_list_B = []
            psnr_list_B = []

    # Save metrics plot and CSV at the end of training
    save_metrics_plot(epoch_mse_A, epoch_psnr_A, epoch_mse_B, epoch_psnr_B, checkpoint_dir)
    save_metrics_csv(epoch_mse_A, epoch_psnr_A, epoch_mse_B, epoch_psnr_B, epoch_losses, checkpoint_dir)
