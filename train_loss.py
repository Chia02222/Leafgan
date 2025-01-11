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
from pytorch_msssim import ssim

def calculate_ssim(real_images, reconstructed_images):
    device = real_images.device
    real_images = real_images.to(device)
    reconstructed_images = reconstructed_images.to(device)
    ssim_score = ssim(real_images, reconstructed_images, data_range=1.0, size_average=True)
    return ssim_score.item()

def calculate_psnr(real_images, reconstructed_images):
    device = real_images.device
    real_images = real_images.to(device)
    reconstructed_images = reconstructed_images.to(device)
    mse = torch.nn.functional.mse_loss(real_images, reconstructed_images)
    psnr = 10 * torch.log10(1 / mse)
    return psnr.item()

def save_metrics_plot(epoch_ssim_A, epoch_psnr_A, epoch_ssim_B, epoch_psnr_B, epoch_losses, checkpoint_dir):
    # Calculate the x-axis labels for 5-epoch intervals
    x_labels = [i * 5 for i in range(len(epoch_ssim_A))]
    
    # Plot metrics
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(x_labels, epoch_ssim_A, label='Average SSIM A')
    plt.xlabel('Epoch (x5)')
    plt.ylabel('Average SSIM A')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(x_labels, epoch_psnr_A, label='Average PSNR A')
    plt.xlabel('Epoch (x5)')
    plt.ylabel('Average PSNR A')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(x_labels, epoch_ssim_B, label='Average SSIM B')
    plt.xlabel('Epoch (x5)')
    plt.ylabel('Average SSIM B')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(x_labels, epoch_psnr_B, label='Average PSNR B')
    plt.xlabel('Epoch (x5)')
    plt.ylabel('Average PSNR B')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'metrics_plot.png'))
    plt.close()

    # Plot epoch losses
    plt.figure()
    plt.plot(x_labels, epoch_losses, label='Epoch Losses')
    plt.xlabel('Epoch (x5)')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'loss_plot.png'))
    plt.close()

def save_metrics_csv(epoch_ssim_A, epoch_psnr_A, epoch_ssim_B, epoch_psnr_B, epoch_losses, checkpoint_dir):
    csv_file = os.path.join(checkpoint_dir, 'metrics.csv')
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Average SSIM A', 'Average PSNR A', 'Average SSIM B', 'Average PSNR B', 'Loss'])
        for epoch in range(len(epoch_ssim_A)):
            writer.writerow([epoch + 1, epoch_ssim_A[epoch], epoch_psnr_A[epoch], epoch_ssim_B[epoch], epoch_psnr_B[epoch], epoch_losses[epoch]])

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = 0

    ssim_list_A = []
    psnr_list_A = []
    ssim_list_B = []
    psnr_list_B = []
    epoch_ssim_A = []
    epoch_psnr_A = []
    epoch_ssim_B = []
    epoch_psnr_B = []
    epoch_losses = []

    best_ssim_A = -float('inf')
    best_psnr_A = -float('inf')
    best_ssim_B = -float('inf')
    best_psnr_B = -float('inf')

    final_ssim_A = None
    final_psnr_A = None
    final_ssim_B = None
    final_psnr_B = None

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

                ssim_A = calculate_ssim(real_A, rec_A)
                psnr_A = calculate_psnr(real_A, rec_A)

                ssim_list_A.append(ssim_A)
                psnr_list_A.append(psnr_A)

                # Track best metrics for A
                if ssim_A > best_ssim_A:
                    best_ssim_A = ssim_A
                if psnr_A > best_psnr_A:
                    best_psnr_A = psnr_A

            if rec_B_key in visuals:
                rec_B = visuals[rec_B_key].to(model.device)

                ssim_B = calculate_ssim(real_B, rec_B)
                psnr_B = calculate_psnr(real_B, rec_B)

                ssim_list_B.append(ssim_B)
                psnr_list_B.append(psnr_B)

                # Track best metrics for B
                if ssim_B > best_ssim_B:
                    best_ssim_B = ssim_B
                if psnr_B > best_psnr_B:
                    best_psnr_B = psnr_B

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

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
            avg_ssim_A = np.mean(ssim_list_A)
            avg_psnr_A = np.mean(psnr_list_A)
            avg_ssim_B = np.mean(ssim_list_B)
            avg_psnr_B = np.mean(psnr_list_B)

            
            epoch_ssim_A.append(avg_ssim_A)
            epoch_psnr_A.append(avg_psnr_A)
            epoch_ssim_B.append(avg_ssim_B)
            epoch_psnr_B.append(avg_psnr_B)
            avg_loss = np.mean([losses[k] for k in losses])  # Calculate average loss
            epoch_losses.append(avg_loss)

            # Store final metrics
            final_ssim_A = avg_ssim_A
            final_psnr_A = avg_psnr_A
            final_ssim_B = avg_ssim_B
            final_psnr_B = avg_psnr_B

            print(f'Epoch: {epoch}, Average SSIM A: {avg_ssim_A:.4f}, Average PSNR A: {avg_psnr_A:.4f}')
            print(f'Epoch: {epoch}, Average SSIM B: {avg_ssim_B:.4f}, Average PSNR B: {avg_psnr_B:.4f}')

        print(f'End of epoch {epoch} / {opt.niter + opt.niter_decay} \t Time Taken: {time.time() - epoch_start_time:.2f} sec')
        model.update_learning_rate()

    save_metrics_plot(epoch_ssim_A, epoch_psnr_A, epoch_ssim_B, epoch_psnr_B, epoch_losses, opt.checkpoints_dir)
    save_metrics_csv(epoch_ssim_A, epoch_psnr_A, epoch_ssim_B, epoch_psnr_B, epoch_losses, opt.checkpoints_dir)
    visualizer.save_log_to_excel("my_training_log.xlsx")

    # Save best and final metrics
    with open(os.path.join(opt.checkpoints_dir, 'best_and_final_metrics.txt'), 'w') as f:
        f.write(f'Best SSIM A: {best_ssim_A:.4f}, Best PSNR A: {best_psnr_A:.4f}\n')
        f.write(f'Best SSIM B: {best_ssim_B:.4f}, Best PSNR B: {best_psnr_B:.4f}\n')
        f.write(f'Final SSIM A: {final_ssim_A:.4f}, Final PSNR A: {final_psnr_A:.4f}\n')
        f.write(f'Final SSIM B: {final_ssim_B:.4f}, Final PSNR B: {final_psnr_B:.4f}\n')
    
