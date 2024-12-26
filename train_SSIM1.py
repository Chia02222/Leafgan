import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import csv
import cv2
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from skimage.metrics import structural_similarity as ssim
from sklearn.decomposition import PCA
from torchvision import transforms  # Add this import for transforms
from PIL import Image

def calculate_ssim(real_image, reconstructed_image):
    real_image = real_image.detach().cpu().numpy().transpose(1, 2, 0)
    reconstructed_image = reconstructed_image.detach().cpu().numpy().transpose(1, 2, 0)

    real_image = (real_image * 255).astype(np.uint8)
    reconstructed_image = (reconstructed_image * 255).astype(np.uint8)

    real_gray = cv2.cvtColor(real_image, cv2.COLOR_RGB2GRAY)
    reconstructed_gray = cv2.cvtColor(reconstructed_image, cv2.COLOR_RGB2GRAY)

    return ssim(real_gray, reconstructed_gray)
    
def calculate_psnr(real_images, reconstructed_images):
    device = real_images.device
    real_images = real_images.to(device)
    reconstructed_images = reconstructed_images.to(device)
    mse = torch.nn.functional.mse_loss(real_images, reconstructed_images)
    psnr = 10 * torch.log10(1 / mse)
    return psnr.item()
    
def save_metrics_plot(epoch_psnr_A, epoch_ssim_A, epoch_psnr_B, epoch_ssim_B,epoch_losses, checkpoint_dir):
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(range(len(epoch_psnr_A)), epoch_psnr_A, label='Average PSNR A')
    plt.xlabel('Epoch')
    plt.ylabel('Average PSNR A')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(range(len(epoch_ssim_A)), epoch_ssim_A, label='Average SSIM A')
    plt.xlabel('Epoch')
    plt.ylabel('Average SSIM A')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(range(len(epoch_psnr_B)), epoch_psnr_B, label='Average psnr B')
    plt.xlabel('Epoch')
    plt.ylabel('Average PSNR B')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(range(len(epoch_ssim_B)), epoch_ssim_B, label='Average SSIM B')
    plt.xlabel('Epoch')
    plt.ylabel('Average SSIM B')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'metrics_plot.png'))
    plt.close()
    
    # Plot epoch losses
    plt.figure()
    plt.plot(range(len(epoch_losses)), epoch_losses, label='Epoch Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'loss_plot.png'))
    plt.close()

def save_metrics_csv(epoch_psnr_A, epoch_ssim_A, epoch_psnr_B, epoch_ssim_B, epoch_losses, checkpoint_dir):
    csv_file = os.path.join(checkpoint_dir, 'metrics.csv')
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Average PSNR A', 'Average SSIM A', 'Average PSNR B', 'Average SSIM B', 'Loss'])
        for epoch in range(len(epoch_fid_A)):
            writer.writerow([epoch + 1, epoch_psnr_A[epoch], epoch_ssim_A[epoch], epoch_psnr_B[epoch], epoch_ssim_B[epoch], epoch_losses[epoch]])

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = 0

    psnr_list_A, ssim_list_A, psnr_list_B, ssim_list_B, epoch_losses = [], [], [], [], []
    
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

            model.compute_visuals()
            visuals = model.get_current_visuals()
            real_A = data['A'].to(model.device)
            rec_A = visuals['rec_A'].to(model.device)
            real_B = data['B'].to(model.device)
            rec_B = visuals['rec_B'].to(model.device)

            if rec_A_key in visuals:
                rec_A = visuals[rec_A_key].to(model.device)
                ssim_A = calculate_ssim(real_A[0], rec_A[0])
                ssim_list_A.append(ssim_A)
                psnr_A = calculate_psnr(real_A, rec_A)
                psnr_list_A.append(psnr_A)
                
                # Track best metrics for A
                if ssim_A > best_ssim_A:
                    best_ssim_A = ssim_A
                if psnr_A > best_psnr_A:
                    best_psnr_A = psnr_A

                if total_iters % opt.print_freq == 0:
                    print(f'SSIM A: {ssim_A}')
                    print(f'PSNR A: {psnr_A}')
                    

            if rec_B_key in visuals:
                rec_B = visuals[rec_B_key].to(model.device)
                ssim_B = calculate_ssim(real_B[0], rec_B[0])
                ssim_list_B.append(ssim_B)
                psnr_B = calculate_psnr(real_B, rec_B)
                psnr_list_B.append(psnr_B)

                # Track best metrics for B
                if ssim_B > best_ssim_B:
                    best_ssim_B = ssim_B
                if psnr_B > best_psnr_B:
                    best_psnr_B = psnr_B

                if total_iters % opt.print_freq == 0:
                    print(f'SSIM B: {ssim_B}')
                    print(f'PSNR B: {psnr_B}')


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

        print(f'Epoch {epoch} - SSIM A: {ssim_A} - SSIM B: {ssim_B}')
            
        if epoch % opt.save_epoch_freq == 0:
            print(f'Saving the model at the end of epoch {epoch}, iters {total_iters}')
            model.save_networks('latest')
            model.save_networks(epoch)
            
            avg_ssim_A = np.mean(ssim_list_A) if ssim_list_A else None
            avg_ssim_B = np.mean(ssim_list_B) if ssim_list_B else None
            avg_psnr_A = np.mean(psnr_list_A) if psnr_list_A else None
            avg_psnr_B = np.mean(psnr_list_B) if psnr_list_B else None
            avg_loss = np.mean([losses[k] for k in losses])
            epoch_losses.append(avg_loss)
            epoch_ssim_A.append(avg_ssim_A)
            epoch_psnr_A.append(avg_psnr_A)
            epoch_ssim_B.append(avg_ssim_B)
            epoch_psnr_B.append(avg_psnr_B)
            epoch_losses.append(avg_loss)

            final_mse_A = avg_mse_A
            final_psnr_A = avg_psnr_A
            final_mse_B = avg_mse_B
            final_psnr_B = avg_psnr_B

            print(f'Epoch {epoch} - Average SSIM A: {avg_ssim_A} - Average SSIM B: {avg_ssim_B}')
            print(f'Epoch {epoch} - Average PSNR A: {avg_psnr_A} - Average PSNR B: {avg_psnr_B}')

        epoch_end_time = time.time()
        print(f'Epoch {epoch} completed in {epoch_end_time - epoch_start_time:.2f} seconds.')

    save_metrics_plot(fid_list_A, ssim_list_A, fid_list_B, ssim_list_B,epoch_looses, checkpoint_dir)
    save_metrics_csv(fid_list_A, ssim_list_A, fid_list_B, ssim_list_B, epoch_losses, checkpoint_dir)

    # Save best and final metrics
    with open(os.path.join(opt.checkpoints_dir, 'best_and_final_metrics.txt'), 'w') as f:
        f.write(f'Best PSNR A: {best_psnr_A:.4f}, Best PSNR A: {best_psnr_A:.4f}\n')
        f.write(f'Best PSNR B: {best_psnr_B:.4f}, Best PSNR B: {best_psnr_B:.4f}\n')
        f.write(f'Final PSNR A: {final_psnr_A:.4f}, Final PSNR A: {final_psnr_A:.4f}\n')
        f.write(f'Final PSNR B: {final_psnr_B:.4f}, Final PSNR B: {final_psnr_B:.4f}\n')

    model.update_learning_rate()
    model.update_learning_rate()
