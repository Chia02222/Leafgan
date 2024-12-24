import os
import time
import numpy as np
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util import calculate_fid, calculate_ssim, save_metrics_plot, save_metrics_csv

if __name__ == '__main__':
    # Parse the training options
    opt = TrainOptions().parse()

    # Create the dataset and initialize the model
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f'The number of training images = {dataset_size}')

    model = create_model(opt)
    model.setup(opt)

    # Initialize visualizer for display and metrics tracking
    visualizer = Visualizer(opt)
    total_iters = 0

    # Initialize lists to track FID and SSIM values
    fid_list_A = []
    ssim_list_A = []
    fid_list_B = []
    ssim_list_B = []
    epoch_fid_A = []
    epoch_ssim_A = []
    epoch_fid_B = []
    epoch_ssim_B = []
    epoch_losses = []

    # Create the checkpoints directory if it doesn't exist
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        # Reset lists for SSIM values at the start of each epoch
        fid_list_A = []
        ssim_list_A = []
        fid_list_B = []
        ssim_list_B = []

        # Iterate through the dataset
        for i, data in enumerate(dataset):
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            # Set model input and optimize parameters
            model.set_input(data)
            model.optimize_parameters()

            # Get the real and reconstructed images
            real_A = data['A'].to(model.device)
            real_B = data['B'].to(model.device)
            visuals = model.get_current_visuals()

            rec_A_key = 'rec_A'
            rec_B_key = 'rec_B'

            # SSIM calculation for image A (only once per batch)
            if rec_A_key in visuals:
                rec_A = visuals[rec_A_key].to(model.device)
                ssim_A = calculate_ssim(real_A[0], rec_A[0])
                ssim_list_A.append(ssim_A)

            # SSIM calculation for image B (only once per batch)
            if rec_B_key in visuals:
                rec_B = visuals[rec_B_key].to(model.device)
                ssim_B = calculate_ssim(real_B[0], rec_B[0])
                ssim_list_B.append(ssim_B)

        # After processing all batches in the epoch, calculate and print SSIM values
        avg_ssim_A = np.mean(ssim_list_A)
        avg_ssim_B = np.mean(ssim_list_B)
        print(f'Epoch {epoch} - Average SSIM A: {avg_ssim_A}')
        print(f'Epoch {epoch} - Average SSIM B: {avg_ssim_B}')

        # FID calculation every 10 epochs
        if epoch % 10 == 0:
            fid_A = calculate_fid(real_A, rec_A, transform)
            fid_B = calculate_fid(real_B, rec_B, transform)

            fid_list_A.append(fid_A)
            fid_list_B.append(fid_B)

            print(f'Epoch {epoch} - FID A: {fid_A}')
            print(f'Epoch {epoch} - FID B: {fid_B}')

        # Calculate and append average metrics for the epoch
        avg_fid_A = np.mean(fid_list_A) if fid_list_A else None
        avg_fid_B = np.mean(fid_list_B) if fid_list_B else None

        epoch_fid_A.append(avg_fid_A)
        epoch_ssim_A.append(avg_ssim_A)
        epoch_fid_B.append(avg_fid_B)
        epoch_ssim_B.append(avg_ssim_B)

        # Save metrics plot and CSV for the epoch
        save_metrics_plot(epoch_fid_A, epoch_ssim_A, epoch_fid_B, epoch_ssim_B, checkpoint_dir)
        save_metrics_csv(epoch_fid_A, epoch_ssim_A, epoch_fid_B, epoch_ssim_B, epoch_losses, checkpoint_dir)

        # Print the average metrics for the epoch
        print(f'Epoch {epoch} - Average FID A: {avg_fid_A if avg_fid_A is not None else "N/A"}')
        print(f'Epoch {epoch} - Average SSIM A: {avg_ssim_A}')
        print(f'Epoch {epoch} - Average FID B: {avg_fid_B if avg_fid_B is not None else "N/A"}')
        print(f'Epoch {epoch} - Average SSIM B: {avg_ssim_B}')
        
        # Optionally, save the model checkpoint after each epoch
        model.save(epoch)
        
        # Print elapsed time for the epoch
        epoch_end_time = time.time()
        print(f'Epoch {epoch} completed in {epoch_end_time - epoch_start_time:.2f} seconds.')
