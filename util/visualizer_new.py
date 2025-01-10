import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)

        util.save_image(im, save_path, aspect_ratio=aspect_ratio)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.epoch_losses = {}
        self.saved = False
        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):
    """
    Display current results on visdom; save current results to an HTML file.

    Parameters:
        visuals (OrderedDict) - - dictionary of images to display or save
        epoch (int) - - the current epoch
        save_result (bool) - - if save the current results to an HTML file
    """
    if self.use_html and (save_result or not self.saved):  # Save images to an HTML file
        self.saved = True

        # 保存特定图片的变化
        if self.target_image_path:  # 如果设置了目标图片路径
            target_name = ntpath.basename(self.target_image_path)
            for label, image in visuals.items():
                if label in target_name:  # 匹配目标图片的标签
                    image_numpy = util.tensor2im(image)

                    # 计算当前 epoch 所属分组，例如 epoch 1-10, 11-20
                    group = (epoch - 1) // 10 + 1
                    group_dir = os.path.join(self.img_dir, f'group_{group * 10 - 9}-{group * 10}')
                    util.mkdirs(group_dir)  # 创建分组文件夹

                    # 保存图片
                    img_path = os.path.join(group_dir, f'epoch{epoch:03d}_{label}.png')
                    util.save_image(image_numpy, img_path)
        else:
            # 默认行为：保存所有图片
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)

                # 计算当前 epoch 所属分组
                group = (epoch - 1) // 10 + 1
                group_dir = os.path.join(self.img_dir, f'group_{group * 10 - 9}-{group * 10}')
                util.mkdirs(group_dir)

                # 保存图片
                img_path = os.path.join(group_dir, f'epoch{epoch:03d}_{label}.png')
                util.save_image(image_numpy, img_path)
              
    def accumulate_loss(self, epoch, losses):
        """Accumulate the losses for the given epoch."""
        if epoch not in self.epoch_losses:
            self.epoch_losses[epoch] = {key: 0.0 for key in losses.keys()}  # Initialize loss accumulator
        
        for k, v in losses.items():
            self.epoch_losses[epoch][k] += v  # Accumulate losses for each key
    
    def save_log_to_excel(self, excel_path="loss_log.xlsx"):
        """Save accumulated training losses to an Excel file."""
        import pandas as pd

        log_data = []
        for epoch, losses in self.epoch_losses.items():
            loss_data = {"epoch": epoch}
            loss_data.update(losses)
            log_data.append(loss_data)

        # Convert the loss data to a DataFrame
        df = pd.DataFrame(log_data)

        # Save the loss data to an Excel file
        df.to_excel(excel_path, index=False)
        print(f"Log saved to Excel file: {excel_path}")

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """Print current losses, accumulate them, and save to the log file."""
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        
        # Accumulate the losses for the current epoch
        self.accumulate_loss(epoch, losses)
        
        # Add loss information for each label
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # Print message to console
        
        # Log the message into the loss log file
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # Save log message
    
