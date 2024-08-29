import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

# Import metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    all_preds = []
    all_labels = []

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
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

            # Get predictions and labels for metrics calculation
            preds = model.get_current_predictions()  # You'll need to implement this method in your model class
            labels = data['label'].cpu().numpy()     # Ensure that the labels are in a compatible format
            
            all_preds.extend(preds)
            all_labels.extend(labels)

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

        # Calculate and print metrics at the end of each epoch
        confusion = confusion_matrix(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        accuracy = accuracy_score(all_labels, all_preds)

        print(f'Confusion Matrix:\n{confusion}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'Accuracy: {accuracy:.4f}')

        # Reset lists for next epoch
        all_preds = []
        all_labels = []

        print(f'End of epoch {epoch} / {opt.niter + opt.niter_decay} \t Time Taken: {time.time() - epoch_start_time:.2f} sec')
        model.update_learning_rate()  # update learning rates at the end of every epoch.
