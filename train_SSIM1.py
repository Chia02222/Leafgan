import time
import numpy as np
import torch
import os
import csv
import cv2
from skimage.metrics import structural_similarity as ssim
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

    epoch_ssim_A = []
    epoch_ssim_B = []

    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        ssim_list_A = []
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
                ssim_A = calculate_ssim(real_A[0], rec_A[0])

                ssim_list_A.append(ssim_A)

                print(f'SSIM A: {ssim_A}')

            if rec_B_key in visuals:
                rec_B = visuals[rec_B_key].to(model.device)
                ssim_B = calculate_ssim(real_B[0], rec_B[0])

                ssim_list_B.append(ssim_B)

                print(f'SSIM B: {ssim_B}')

        avg_ssim_A = np.mean(ssim_list_A)
        avg_ssim_B = np.mean(ssim_list_B)

        epoch_ssim_A.append(avg_ssim_A)
        epoch_ssim_B.append(avg_ssim_B)

        print(f'Epoch: {epoch}, Average SSIM A: {avg_ssim_A:.4f}')
        print(f'Epoch: {epoch}, Average SSIM B: {avg_ssim_B:.4f}')

        print(f'End of epoch {epoch} / {opt.niter + opt.niter_decay} \t Time Taken: {time.time() - epoch_start_time:.2f} sec')

    save_metrics_plot(epoch_ssim_A, epoch_ssim_B, opt.checkpoints_dir)
    save_metrics_csv(epoch_ssim_A, epoch_ssim_B, [], opt.checkpoints_dir)
