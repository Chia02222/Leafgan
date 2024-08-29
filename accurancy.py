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
            preds = model.get_current_predictions()  # Ensure this method is implemented in your model class
            labels = data['label'].cpu().numpy()     # Ensure that the labels are in a compatible format
            
            all_preds.extend(preds)
            all_labels.extend(labels)

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
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

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

try:
    opt = TrainOptions().parse()   # get training options
except Exception as e:
    print(f"Error in parsing options: {e}")
    raise

# Ensure data loading is correct
print(f"Dataset size: {dataset_size}")

for i, data in enumerate(dataset):  # inner loop within one epoch
    try:
        model.set_input(data)
    except Exception as e:
        print(f"Error setting input: {e}")
        raise
    
    try:
        model.optimize_parameters()
    except Exception as e:
        print(f"Error optimizing parameters: {e}")
        raise
    
    try:
        preds = model.get_current_predictions()
        print(f"Predictions: {preds}")
    except Exception as e:
        print(f"Error getting predictions: {e}")
        raise
    
    try:
        labels = data['label'].cpu().numpy()
        print(f"Labels: {labels}")
    except Exception as e:
        print(f"Error processing labels: {e}")
        raise
