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
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim
from sklearn.decomposition import PCA
import cv2
from PIL import Image

def calculate_fid(real_images, reconstructed_images, transform, batch_size=8, pca_components=5):
    device = real_images.device
    real_images = real_images.to(device)
    reconstructed_images = reconstructed_images.to(device)

    # Check for sufficient samples
    if len(real_images) < 10 or len(reconstructed_images) < 10:
        print("Not enough samples for FID calculation. Need at least 10.")
        return 0

    real_features = []
    reconstructed_features = []

    for i in range(0, len(real_images), batch_size):
        batch_real = real_images[i:i + batch_size]
        batch_rec = reconstructed_images[i:i + batch_size]

        batch_real_features = []
        batch_rec_features = []

        for img_real, img_rec in zip(batch_real, batch_rec):
            img_real_pil = Image.fromarray((img_real.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))
            img_rec_pil = Image.fromarray((img_rec.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))

            img_real_transformed = transform(img_real_pil).unsqueeze(0).to(device)
            img_rec_transformed = transform(img_rec_pil).unsqueeze(0).to(device)

            batch_real_features.append(img_real_transformed)
            batch_rec_features.append(img_rec_transformed)

        real_features.extend(batch_real_features)
        reconstructed_features.extend(batch_rec_features)

    real_features = torch.cat(real_features, dim=0).view(-1, real_features[0].numel()).cpu().numpy()
    reconstructed_features = torch.cat(reconstructed_features, dim=0).view(-1, reconstructed_features[0].numel()).cpu().numpy()

    # Apply PCA and calculate FID
    pca = PCA(n_components=pca_components)
    real_features_pca = pca.fit_transform(real_features)
    reconstructed_features_pca = pca.transform(reconstructed_features)

    mu1, sigma1 = np.mean(real_features_pca, axis=0), np.cov(real_features_pca, rowvar=False)
    mu2, sigma2 = np.mean(reconstructed_features_pca, axis=0), np.cov(reconstructed_features_pca, rowvar=False)

    epsilon = 1e-6
    sigma1 += epsilon * np.eye(sigma1.shape[0])
    sigma2 += epsilon * np.eye(sigma2.shape[0])

    fid = np.sum((mu1 - mu2) ** 2) + np.trace(sigma1 + sigma2 - 2 * sqrtm(sigma1 @ sigma2))
    return fid

def calculate_ssim(real_image, reconstructed_image):
    # Detach the tensors and convert to NumPy for SSIM calculation
    real_image = real_image.detach().cpu().numpy().transpose(1, 2, 0)  # Convert to HxWxC
    reconstructed_image = reconstructed_image.detach().cpu().numpy().transpose(1, 2, 0)
    
    # Convert to 8-bit format for SSIM calculation
    real_image = (real_image * 255).astype(np.uint8)
    reconstructed_image = (reconstructed_image * 255).astype(np.uint8)

    # Convert to grayscale
    real_gray = cv2.cvtColor(real_image, cv2.COLOR_RGB2GRAY)
    reconstructed_gray = cv2.cvtColor(reconstructed_image, cv2.COLOR_RGB2GRAY)

    # Calculate SSIM between grayscale images
    return ssim(real_gray, reconstructed_gray)

def save_metrics_plot(epoch_fid_A, epoch_ssim_A, epoch_fid_B, epoch_ssim_B, checkpoint_dir):
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(range(len(epoch_fid_A)), epoch_fid_A, label='Average FID A')
    plt.xlabel('Epoch')
    plt.ylabel('Average FID A')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(range(len(epoch_ssim_A)), epoch_ssim_A, label='Average SSIM A')
    plt.xlabel('Epoch')
    plt.ylabel('Average SSIM A')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(range(len(epoch_fid_B)), epoch_fid_B, label='Average FID B')
    plt.xlabel('Epoch')
    plt.ylabel('Average FID B')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(range(len(epoch_ssim_B)), epoch_ssim_B, label='Average SSIM B')
    plt.xlabel('Epoch')
    plt.ylabel('Average SSIM B')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'metrics_plot.png'))
    plt.close()

def save_metrics_csv(epoch_fid_A, epoch_ssim_A, epoch_fid_B, epoch_ssim_B, epoch_losses, checkpoint_dir):
    csv_file = os.path.join(checkpoint_dir, 'metrics.csv')
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Average FID A', 'Average SSIM A', 'Average FID B', 'Average SSIM B', 'Loss'])
        for epoch in range(len(epoch_fid_A)):
            writer.writerow([epoch + 1, epoch_fid_A[epoch], epoch_ssim_A[epoch], epoch_fid_B[epoch], epoch_ssim_B[epoch], epoch_losses[epoch]])
            
if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualizer that displays/saves images and plots
    total_iters = 0  # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):  # outer loop for different epochs
        epoch_start_time = time.time()  # timer for the entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in the current epoch

        # Initialize FID/SSIM lists for tracking metrics
        fid_list_A = []
        ssim_list_A = []
        fid_list_B = []
        ssim_list_B = []
        
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            # Optionally display the results
            if total_iters % opt.display_freq == 0:  # display images on visdom and save images to an HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            # Optionally print current losses
            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            # Save the latest model every <save_latest_freq> iterations
            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        # Cache the model at the end of every epoch
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        # Calculate FID and SSIM for monitoring progress
        if epoch % 10 == 0:  # Calculate FID/SSIM every 10 epochs (or any preferred interval)
            real_A = data['A'].to(model.device)
            rec_A = visuals['rec_A'].to(model.device)
            real_B = data['B'].to(model.device)
            rec_B = visuals['rec_B'].to(model.device)

            # Compute SSIM (example function provided earlier in the script)
            ssim_A = calculate_ssim(real_A[0], rec_A[0])
            ssim_B = calculate_ssim(real_B[0], rec_B[0])
            fid_A = calculate_fid(real_A, rec_A, transform)
            fid_B = calculate_fid(real_B, rec_B, transform)

            ssim_list_A.append(ssim_A)
            ssim_list_B.append(ssim_B)
            fid_list_A.append(fid_A)
            fid_list_B.append(fid_B)

            print(f'Epoch {epoch} - SSIM A: {ssim_A} - SSIM B: {ssim_B} - FID A: {fid_A} - FID B: {fid_B}')

        # Average FID and SSIM for the current epoch
        avg_ssim_A = np.mean(ssim_list_A) if ssim_list_A else None
        avg_ssim_B = np.mean(ssim_list_B) if ssim_list_B else None
        avg_fid_A = np.mean(fid_list_A) if fid_list_A else None
        avg_fid_B = np.mean(fid_list_B) if fid_list_B else None

        print(f'Epoch {epoch} - Average SSIM A: {avg_ssim_A} - Average SSIM B: {avg_ssim_B}')
        print(f'Epoch {epoch} - Average FID A: {avg_fid_A} - Average FID B: {avg_fid_B}')

        # Optionally, save FID/SSIM plot and CSV
        save_metrics_plot(fid_list_A, ssim_list_A, fid_list_B, ssim_list_B, checkpoint_dir)
        save_metrics_csv(fid_list_A, ssim_list_A, fid_list_B, ssim_list_B, checkpoint_dir)

        epoch_end_time = time.time()
        print(f'Epoch {epoch} completed in {epoch_end_time - epoch_start_time:.2f} seconds.')

        # Update learning rates at the end of every epoch
        model.update_learning_rate()

     
