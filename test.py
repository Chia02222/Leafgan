import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
import csv
from pytorch_fid import fid_score
from pytorch_gan_metrics import get_inception_score
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

def calculate_rmse(imageA, imageB):
    mse = compare_mse(imageA, imageB)
    rmse = np.sqrt(mse)
    return rmse

def calculate_psnr(imageA, imageB):
    psnr = compare_psnr(imageA, imageB)
    return psnr

class ImageDataset(Dataset):
    def __init__(self, images):
        self.images = images
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].numpy().transpose(1, 2, 0)
        img = Image.fromarray((img * 255).astype(np.uint8))
        img = self.transform(img)
        return img

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    # Initialize CSV file to save metrics
    metrics_file = os.path.join(web_dir, 'metrics.csv')
    with open(metrics_file, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'RMSE', 'PSNR'])

    if opt.eval:
        model.eval()

    real_images = []
    generated_images = []

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        
        # Convert tensors to numpy arrays for RMSE and PSNR calculation
        real_A = visuals['real_A'].cpu().numpy().transpose(1, 2, 0)  # Convert tensor to numpy
        fake_B = visuals['fake_B'].cpu().numpy().transpose(1, 2, 0)  # Convert tensor to numpy

        # Save real and generated images for FID/IS calculation
        real_images.append(visuals['real_A'].cpu())
        generated_images.append(visuals['fake_B'].cpu())

        # Calculate RMSE and PSNR
        rmse_value = calculate_rmse(real_A, fake_B)
        psnr_value = calculate_psnr(real_A, fake_B)

        print(f'Image {i}: RMSE = {rmse_value}, PSNR = {psnr_value}')
        
        # Save metrics to CSV
        with open(metrics_file, mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([img_path[0], rmse_value, psnr_value])
        
        # Save the images
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    webpage.save()  # save the HTML

    # Convert list of tensors to datasets
    real_dataset = ImageDataset(real_images)
    generated_dataset = ImageDataset(generated_images)

    # Create data loaders
    real_loader = DataLoader(real_dataset, batch_size=1, shuffle=False)
    generated_loader = DataLoader(generated_dataset, batch_size=1, shuffle=False)

    # Calculate FID
    fid_value = fid_score.calculate_fid_given_paths(real_loader, generated_loader, batch_size=1, device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f'FID: {fid_value}')

    # Calculate Inception Score
    is_value, is_std = get_inception_score(generated_loader, device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Inception Score: {is_value}, Standard Deviation: {is_std}')

    # Save FID and IS values to CSV
    with open(metrics_file, mode='a') as file:
        writer = csv.writer(file)
        writer.writerow(['FID', fid_value])
        writer.writerow(['Inception Score', is_value])
        writer.writerow(['Inception Score Std Dev', is_std])
