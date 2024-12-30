import itertools
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from util.image_pool import ImagePool

from . import networks
from .base_model import BaseModel
from .grad_cam import GradCAM

from skimage.metrics import structural_similarity as ssim


class LeafGANModel(BaseModel):
    """
    This class implements the LeafGAN model, for generating high-quality and diversity disease images from healthy.
    LeafGAN is basically an improved version of CycleGAN with the attention mechanism to focus on translating leaf area only.
    LeafGAN paper: https://arxiv.org/abs/2002.10100
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options."""
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping.')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class."""
        BaseModel.__init__(self, opt)
        self.is_using_mask = opt.dataset_mode == "unaligned_masked"
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')
        self.visual_names = visual_names_A + visual_names_B

        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:
            self.model_names = ['G_A', 'G_B']

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                                            opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                                            opt.init_type, opt.init_gain, self.gpu_ids)
            if not self.is_using_mask:
                self.segResNet = models.resnet101()
                num_ftrs = self.segResNet.fc.in_features
                self.segResNet.fc = nn.Linear(num_ftrs, 3)
                load_path = "../LFLSeg/LFLSeg_resnet101.pth"
                self.segResNet.load_state_dict(torch.load(load_path), strict=True)
                self.segResNet.to(self.device)
                self.segResNet.eval()
                self.netLFLSeg = GradCAM(model=self.segResNet)

            if opt.lambda_identity > 0.0:
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionBackground = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def forward(self):
        """Run forward pass."""
        if self.isTrain:
            if not self.is_using_mask:
                self.background_real_A, self.foreground_real_A = self.get_masking(self.real_A, self.opt.threshold)
                self.background_real_B, self.foreground_real_B = self.get_masking(self.real_B, self.opt.threshold)
            self.fake_B = self.netG_A(self.real_A)
            self.rec_A = self.netG_B(self.fake_B)
            self.fake_A = self.netG_B(self.real_B)
            self.rec_B = self.netG_A(self.fake_A)
        else:
            self.fake_B = self.netG_A(self.real_A)
            self.rec_A = self.netG_B(self.fake_B)
            self.fake_A = self.netG_B(self.real_B)
            self.rec_B = self.netG_A(self.fake_A)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights."""
        self.forward()
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()

    def calculate_accuracy(self):
        """Calculate and print the MSE, PSNR, and SSIM between real and reconstructed images."""
        mse = self.calculate_mse(self.real_A, self.rec_A)
        psnr = self.calculate_psnr(self.real_A, self.rec_A)
        ssim_value = self.calculate_ssim(self.real_A, self.rec_A)
        
        print(f'PSNR: {psnr}')
        print(f'SSIM: {ssim_value}')
    
    def calculate_psnr(self, real_images, reconstructed_images):
        """Calculate Peak Signal-to-Noise Ratio (PSNR)."""
        mse = torch.nn.functional.mse_loss(real_images, reconstructed_images)
        psnr = 10 * torch.log10(1 / mse)
        return psnr.item()
    
    def calculate_ssim(self, real_images, reconstructed_images):
        """Calculate Structural Similarity Index Measure (SSIM)."""
        real_images = real_images.cpu().numpy().transpose(0, 2, 3, 1)  # Convert to HWC
        reconstructed_images = reconstructed_images.cpu().numpy().transpose(0, 2, 3, 1)  # Convert to HWC
        ssim_scores = [ssim(real, rec, multichannel=True) for real, rec in zip(real_images, reconstructed_images)]
        return np.mean(ssim_scores)
