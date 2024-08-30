import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import csv
from options.test_options import TestOptions  
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

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
    if mse == 0:
        return float('inf')  # Avoid division by zero
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

def save_metrics_csv(epoch_mse_A, epoch_psnr_A, epoch_mse_B, epoch_psnr_B, checkpoint_dir):
    csv_file = os.path.join(checkpoint_dir, 'metrics.csv')
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Average MSE A', 'Average PSNR A', 'Average MSE B', 'Average PSNR B'])
        for epoch in range(len(epoch_mse_A)):
            writer.writerow([epoch + 1, epoch_mse_A[epoch], epoch_psnr_A[epoch], epoch_mse_B[epoch], epoch_psnr_B[epoch]])

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    mse_list_A = []
    psnr_list_A = []
    mse_list_B = []
    psnr_list_B = []
    epoch_mse_A = []
    epoch_psnr_A = []
    epoch_mse_B = []
    epoch_psnr_B = []

    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    if opt.eval:
        model.eval()

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        model.set_input(data)
        model.test()  # Run inference

        real_A = data['A'].to(model.device)
        real_B = data['B'].to(model.device)
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths() 
        
        if 'rec_A' in visuals:
            rec_A = visuals['rec_A'].to(model.device)
            mse_A = calculate_mse(real_A, rec_A)
            psnr_A = calculate_psnr(real_A, rec_A)
            mse_list_A.append(mse_A)
            psnr_list_A.append(psnr_A)

        if 'rec_B' in visuals:
            rec_B = visuals['rec_B'].to(model.device)
            mse_B = calculate_mse(real_B, rec_B)
            psnr_B = calculate_psnr(real_B, rec_B)
            mse_list_B.append(mse_B)
            psnr_list_B.append(psnr_B)

        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break

        # Calculate and store average metrics for the epoch
        avg_mse_A = np.mean(mse_list_A) 
        avg_psnr_A = np.mean(psnr_list_A) 
        avg_mse_B = np.mean(mse_list_B) 
        avg_psnr_B = np.mean(psnr_list_B) 
        epoch_mse_A.append(avg_mse_A)
        epoch_psnr_A.append(avg_psnr_A)
        epoch_mse_B.append(avg_mse_B)
        epoch_psnr_B.append(avg_psnr_B)

        print(f'Epoch: {i},  MSE A: {avg_mse_A:.4f}, PSNR A: {avg_psnr_A:.4f}')
        print(f'Epoch: {i},  MSE B: {avg_mse_B:.4f}, PSNR B: {avg_psnr_B:.4f}')

        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    # After looping through the dataset
    checkpoint_dir = os.path.join(opt.results_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_metrics_plot(epoch_mse_A, epoch_psnr_A, epoch_mse_B, epoch_psnr_B, checkpoint_dir)
    save_metrics_csv(epoch_mse_A, epoch_psnr_A, epoch_mse_B, epoch_psnr_B, checkpoint_dir)

    webpage.save()
