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

def save_loss_plot(epoch_losses, checkpoint_dir):
    # Calculate the x-axis labels for 5-epoch intervals
    x_labels = [i * 5 for i in range(len(epoch_losses))]
    
    # Plot epoch losses
    plt.figure()
    plt.plot(x_labels, epoch_losses, label='Epoch Losses')
    plt.xlabel('Epoch (x5)')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'loss_plot.png'))
    plt.close()

def save_loss_csv(epoch_losses, checkpoint_dir):
    csv_file = os.path.join(checkpoint_dir, 'loss.csv')
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Loss'])
        for epoch in range(len(epoch_losses)):
            writer.writerow([epoch + 1, epoch_losses[epoch]])

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = 0
    epoch_losses = []

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

            if rec_B_key in visuals:
                rec_B = visuals[rec_B_key].to(model.device)


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

            avg_loss = np.mean([losses[k] for k in losses])  # Calculate average loss
            epoch_losses.append(avg_loss)

        print(f'End of epoch {epoch} / {opt.niter + opt.niter_decay} \t Time Taken: {time.time() - epoch_start_time:.2f} sec')
        model.update_learning_rate()
        
    save_loss_csv(epoch_losses, opt.checkpoints_dir)
    save_loss_plot(epoch_losses, opt.checkpoints_dir)
    visualizer.save_log_to_excel("my_training_log.xlsx")


