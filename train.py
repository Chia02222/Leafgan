import time
import numpy as np
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

# Import metrics
from skimage.metrics import structural_similarity as ssim

def calculate_mse(real_images, reconstructed_images):
    device = real_images.device  # Get the device of the first tensor
    real_images = real_images.to(device)
    reconstructed_images = reconstructed_images.to(device)
    mse = torch.nn.functional.mse_loss(real_images, reconstructed_images)
    return mse.item()

def calculate_psnr(real_images, reconstructed_images):
    device = real_images.device  # Get the device of the first tensor
    real_images = real_images.to(device)
    reconstructed_images = reconstructed_images.to(device)
    mse = torch.nn.functional.mse_loss(real_images, reconstructed_images)
    psnr = 10 * torch.log10(1 / mse)
    return psnr.item()

def calculate_ssim(real_images, reconstructed_images):
    # Ensure the images are on CPU and detach from the computation graph
    real_images = real_images.detach().cpu().numpy().transpose(0, 2, 3, 1)  # Convert to HWC
    reconstructed_images = reconstructed_images.detach().cpu().numpy().transpose(0, 2, 3, 1)  # Convert to HWC
    ssim_scores = [ssim(real, rec, multichannel=True) for real, rec in zip(real_images, reconstructed_images)]
    return np.mean(ssim_scores)

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            # Calculate and print metrics for each batch
            real_A = data['A'].to(model.device)  # Move to the same device as the model
            
            # Print available keys in visuals to identify the correct key
            visuals = model.get_current_visuals()
            print("Available keys in visuals:", visuals.keys())
            
            rec_A_key = 'rec_A'  # Updated to match the provided get_current_visuals method
            if rec_A_key in visuals:
                rec_A = visuals[rec_A_key].to(model.device)  # Move to the same device as the model

                mse = calculate_mse(real_A, rec_A)
                psnr = calculate_psnr(real_A, rec_A)
                ssim_value = calculate_ssim(real_A, rec_A)
                
                if total_iters % opt.print_freq == 0:
                    print(f'MSE: {mse}')
                    print(f'PSNR: {psnr}')
                    print(f'SSIM: {ssim_value}')
            else:
                print(f"Key '{rec_A_key}' not found in visuals")

            # Display and save images
            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            # Print losses
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                print(f'Epoch: {epoch}, Iteration: {epoch_iter}, Losses: {losses}, Time per batch: {t_comp:.4f}s, Data loading time: {t_data:.4f}s')

            # Save latest model
            if total_iters % opt.save_latest_freq == 0:
                print(f'Saving the latest model (epoch {epoch}, total_iters {total_iters})')
                save_suffix = f'iter_{total_iters}' if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        # Save model at the end of the epoch
        if epoch % opt.save_epoch_freq == 0:
            print(f'Saving the model at the end of epoch {epoch}, iters {total_iters}')
            model.save_networks('latest')
            model.save_networks(epoch)

        print(f'End of epoch {epoch} / {opt.niter + opt.niter_decay} \t Time Taken: {time.time() - epoch_start_time:.2f} sec')
        model.update_learning_rate()  # update learning rates at the end of every epoch.
