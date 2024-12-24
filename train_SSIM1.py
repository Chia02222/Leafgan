import time
import numpy as np
import torch
import os
import csv
import cv2
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image.fid import FrechetInceptionDistance
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

# Define the function for calculating SSIM
def calculate_ssim(real_image, reconstructed_image):
    real_image = real_image.cpu().numpy().transpose(1, 2, 0)  # Convert to HxWxC
    reconstructed_image = reconstructed_image.cpu().numpy().transpose(1, 2, 0)
    real_image = (real_image * 255).astype(np.uint8)
    reconstructed_image = (reconstructed_image * 255).astype(np.uint8)
    real_gray = cv2.cvtColor(real_image, cv2.COLOR_RGB2GRAY)
    reconstructed_gray = cv2.cvtColor(reconstructed_image, cv2.COLOR_RGB2GRAY)
    return ssim(real_gray, reconstructed_gray)


# Define the function for calculating FID
def calculate_fid(real_images, reconstructed_images, fid_metric, batch_size=8):
    """
    Calculate the FID score using torchmetrics' FrechetInceptionDistance.
    """
    # Convert the image tensors to uint8 and scale to [0, 255]
    real_images = (real_images * 255).clamp(0, 255).to(torch.uint8)
    reconstructed_images = (reconstructed_images * 255).clamp(0, 255).to(torch.uint8)

    # Ensure the images are of dtype uint8
    if real_images.dtype != torch.uint8:
        raise ValueError(f"Expected real_images to be of dtype torch.uint8, but got {real_images.dtype}")
    if reconstructed_images.dtype != torch.uint8:
        raise ValueError(f"Expected reconstructed_images to be of dtype torch.uint8, but got {reconstructed_images.dtype}")

    # Print the dtype and shape of the images to verify
    print(f"real_images dtype: {real_images.dtype}, shape: {real_images.shape}")
    print(f"reconstructed_images dtype: {reconstructed_images.dtype}, shape: {reconstructed_images.shape}")

    # Update the FID metric with the images
    fid_metric.update(real_images, real=True)
    fid_metric.update(reconstructed_images, real=False)

    # Compute and return the FID score
    return fid_metric.compute()

# Initialize the FID metric
fid_metric = FrechetInceptionDistance()

# Inside your training loop:
if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = 0

    epoch_fid_A = []
    epoch_ssim_A = []
    epoch_fid_B = []
    epoch_ssim_B = []

    checkpoint_dir = 'checkpoints'
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
                fid_A = calculate_fid(real_A, rec_A, fid_metric)
                ssim_A = calculate_ssim(real_A[0], rec_A[0])

                fid_list_A.append(fid_A)
                ssim_list_A.append(ssim_A)

                print(f'FID A: {fid_A}')
                print(f'SSIM A: {ssim_A}')

            if rec_B_key in visuals:
                rec_B = visuals[rec_B_key].to(model.device)
                fid_B = calculate_fid(real_B, rec_B, fid_metric)
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

        print(f'Epoch: {epoch}, Average FID A: {avg_fid_A:.4f}, Average SSIM A: {avg_ssim_A:.4f}')
        print(f'Epoch: {epoch}, Average FID B: {avg_fid_B:.4f}, Average SSIM B: {avg_ssim_B:.4f}')

        print(f'End of epoch {epoch} / {opt.niter + opt.niter_decay} \t Time Taken: {time.time() - epoch_start_time:.2f} sec')

    save_metrics_plot(epoch_fid_A, epoch_ssim_A, epoch_fid_B, epoch_ssim_B, opt.checkpoints_dir)
    save_metrics_csv(epoch_fid_A, epoch_ssim_A, epoch_fid_B, epoch_ssim_B, [], opt.checkpoints_dir)


remove the fid
