import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import csv
import os
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
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    mse_list_A = []
    psnr_list_A = []
    mse_list_B = []
    psnr_list_B = []
    epoch_mse_A = []
    epoch_psnr_A = []
    epoch_mse_B = []
    epoch_psnr_B = []
    epoch_losses = []
    
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
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

            if rec_B_key in visuals:
                rec_B = visuals[rec_B_key].to(model.device)

                mse_B = calculate_mse(real_B, rec_B)
                psnr_B = calculate_psnr(real_B, rec_B)

                mse_list_B.append(mse_B)
                psnr_list_B.append(psnr_B)

                if total_iters % opt.print_freq == 0:
                    print(f'MSE B: {mse_B}')
                    print(f'PSNR B: {psnr_B}')

        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))

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

            print(f'Epoch: {epoch}, Average MSE A: {avg_mse_A:.4f}, Average PSNR A: {avg_psnr_A:.4f}')
            print(f'Epoch: {epoch}, Average MSE B: {avg_mse_B:.4f}, Average PSNR B: {avg_psnr_B:.4f}')
            
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        save_metrics_plot(epoch_mse_A, epoch_psnr_A, epoch_mse_B, epoch_psnr_B, checkpoint_dir)
        save_metrics_csv(epoch_mse_A, epoch_psnr_A, epoch_mse_B, epoch_psnr_B, epoch_losses, checkpoint_dir)

    webpage.save()  # save the HTML
