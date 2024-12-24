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
from torchvision import models, transforms
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim
import cv2
from PIL import Image
from sklearn.decomposition import PCA
import torch

def calculate_fid(real_images, reconstructed_images, transform, batch_size=8, pca_components=50):
    device = real_images.device
    real_images = real_images.to(device)
    reconstructed_images = reconstructed_images.to(device)

    # Convert tensors to numpy and then to PIL images for FID computation
    real_features = []
    reconstructed_features = []

    # Define the transformation pipeline for FID computation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to a smaller size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Loop over batches to avoid OOM errors
    for i in range(0, len(real_images), batch_size):
        real_batch = real_images[i:i+batch_size]
        rec_batch = reconstructed_images[i:i+batch_size]
        
        batch_real_features = []
        batch_rec_features = []
        
        for img_real, img_reconstructed in zip(real_batch, rec_batch):
            # Convert to numpy arrays and then to PIL images
            img_real_pil = Image.fromarray((img_real.detach().permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))
            img_reconstructed_pil = Image.fromarray((img_reconstructed.detach().permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))

            # Apply transformations
            img_real = transform(img_real_pil).unsqueeze(0).to(device)
            img_reconstructed = transform(img_reconstructed_pil).unsqueeze(0).to(device)

            # Extract features for FID calculation
            batch_real_features.append(img_real)
            batch_rec_features.append(img_reconstructed)

        # Stack features and calculate FID for the batch
        real_features.extend(batch_real_features)
        reconstructed_features.extend(batch_rec_features)

    # Stack all features after batch processing
    real_features = torch.cat(real_features, dim=0)
    reconstructed_features = torch.cat(reconstructed_features, dim=0)

    # Flatten the features (batch_size, channels, height, width) -> (batch_size, channels * height * width)
    real_features = real_features.view(real_features.size(0), -1).cpu().numpy()
    reconstructed_features = reconstructed_features.view(reconstructed_features.size(0), -1).cpu().numpy()

    # Ensure there are enough samples for FID calculation
    if len(real_features) < 2 or len(reconstructed_features) < 2:
        print("Insufficient samples for FID calculation. Returning a default score of 0.")
        return 0

    # Use PCA to reduce dimensionality if needed
    pca = PCA(n_components=pca_components)
    real_features_pca = pca.fit_transform(real_features)
    reconstructed_features_pca = pca.transform(reconstructed_features)

    # Compute means and covariance
    mu1, sigma1 = np.mean(real_features_pca, axis=0), np.cov(real_features_pca, rowvar=False)
    mu2, sigma2 = np.mean(reconstructed_features_pca, axis=0), np.cov(reconstructed_features_pca, rowvar=False)

    # Handle the case where covariance matrix is degenerate (small batch size)
    epsilon = 1e-6
    sigma1 += epsilon * np.eye(sigma1.shape[0])
    sigma2 += epsilon * np.eye(sigma2.shape[0])

    # FID calculation
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
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = 0

    fid_list_A = []
    ssim_list_A = []
    fid_list_B = []
    ssim_list_B = []
    epoch_fid_A = []
    epoch_ssim_A = []
    epoch_fid_B = []
    epoch_ssim_B = []
    epoch_losses = []

    checkpoint_dir = './checkpoints/leafSpot_leafGAN'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        fid_list_A = []
        ssim_list_A = []
        fid_list_B = []
        ssim_list_B = []

        for i, data in enumerate(dataset):
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

                fid_A = calculate_fid(real_A, rec_A, transforms)
                ssim_A = calculate_ssim(real_A[0], rec_A[0])

                fid_list_A.append(fid_A)
                ssim_list_A.append(ssim_A)

                print(f'FID A: {fid_A}')
                print(f'SSIM A: {ssim_A}')

            if rec_B_key in visuals:
                rec_B = visuals[rec_B_key].to(model.device)

                fid_B = calculate_fid(real_B, rec_B, transforms)
                ssim_B = calculate_ssim(real_B[0], rec_B[0])

                fid_list_B.append(fid_B)
                ssim_list_B.append(ssim_B)

                print(f'FID B: {fid_B}')
                print(f'SSIM B: {ssim_B}')

        avg_fid_A = np.mean(fid_list_A)
        avg_ssim_A = np.mean(ssim_list_A)
        avg_fid_B = np.mean(fid_list_B)
        avg_ssim_B = np.mean(ssim_list_B)

        epoch_fid_A.append(avg_fid_A)
        epoch_ssim_A.append(avg_ssim_A)
        epoch_fid_B.append(avg_fid_B)
        epoch_ssim_B.append(avg_ssim_B)

        # Save metrics plot and CSV
        save_metrics_plot(epoch_fid_A, epoch_ssim_A, epoch_fid_B, epoch_ssim_B, checkpoint_dir)
        save_metrics_csv(epoch_fid_A, epoch_ssim_A, epoch_fid_B, epoch_ssim_B, epoch_losses, checkpoint_dir)

        print(f'Epoch {epoch}, FID A: {avg_fid_A}, SSIM A: {avg_ssim_A}, FID B: {avg_fid_B}, SSIM B: {avg_ssim_B}')
