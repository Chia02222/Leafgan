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


class LeafGANModel(BaseModel):
    """
    This class implements the LeafGAN model with perceptual loss added.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='weight for identity mapping loss')
            parser.add_argument('--lambda_perceptual', type=float, default=1.0, help='weight for perceptual loss')  # New argument
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.is_using_mask = opt.dataset_mode == "unaligned_masked"
        
        # Updated loss names to include perceptual losses
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'perc_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'perc_B']
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

        # Define networks
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
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

            # Define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionBackground = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionPerceptual = torch.nn.MSELoss()  # Perceptual loss using MSE

            # Optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # Feature storage for perceptual loss
            self.features_real_A = []
            self.features_fake_A = []
            self.features_real_B = []
            self.features_fake_B = []

    def to_numpy(self, tensor):
        img = tensor.data
        image_numpy = img[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) - 1.0) / 2.0 * 255.0
        image_numpy = image_numpy.astype(np.uint8)
        return image_numpy

    def save_image(self, tensor, filename):
        image_pil = Image.fromarray(self.to_numpy(tensor))
        image_pil.save(filename)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        if self.is_using_mask:
            self.foreground_real_A = input["mask_A" if AtoB else "mask_B"].to(self.device)
            self.foreground_real_B = input["mask_B" if AtoB else "mask_A"].to(self.device)
            with torch.no_grad():
                self.background_real_A = torch.absolute(1.0 - self.foreground_real_A)
                self.background_real_B = torch.absolute(1.0 - self.foreground_real_B)

    def forward(self):
        if self.isTrain:
            if not self.is_using_mask:
                self.background_real_A, self.foreground_real_A = self.get_masking(self.real_A, self.opt.threshold)
                self.background_real_B, self.foreground_real_B = self.get_masking(self.real_B, self.opt.threshold)
            self.fake_B = self.netG_A(self.real_A)
            self.fore_fake_B = self.foreground_real_A * self.fake_B
            self.back_fake_B = self.background_real_A * self.fake_B
            self.fore_real_B = self.foreground_real_B * self.real_B
            self.back_real_B = self.background_real_B * self.real_B
            self.rec_A = self.netG_B(self.fake_B)
            self.fake_A = self.netG_B(self.real_B)
            self.fore_fake_A = self.foreground_real_B * self.fake_A
            self.back_fake_A = self.background_real_B * self.fake_A
            self.fore_real_A = self.foreground_real_A * self.real_A
            self.back_real_A = self.background_real_A * self.real_A
            self.rec_B = self.netG_A(self.fake_A)
        else:
            self.fake_B = self.netG_A(self.real_A)
            self.rec_A = self.netG_B(self.fake_B)
            self.fake_A = self.netG_B(self.real_B)
            self.rec_B = self.netG_A(self.fake_A)

    def hook_features(self, module, input, output):
        """Hook to collect features from discriminator layers."""
        self.features.append(output)

    def register_hooks(self, netD):
        """Register hooks to collect features from all convolutional layers."""
        self.features = []
        for name, module in netD.named_modules():
            if isinstance(module, nn.Conv2d):
                module.register_forward_hook(self.hook_features)

    def compute_perceptual_loss(self, features_real, features_fake):
        """Compute perceptual loss between real and fake features."""
        if not features_real or not features_fake:
            return 0.0
        perceptual_loss = 0.0
        for f_real, f_fake in zip(features_real, features_fake):
            perceptual_loss += self.criterionPerceptual(f_real, f_fake) / len(features_real)
        return perceptual_loss

    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fore_fake_B = self.fake_B_pool.query(self.fore_fake_B)
        self.register_hooks(self.netD_A)
        self.netD_A(self.fore_real_B)  # Collect real features
        self.features_real_B = self.features.copy()
        self.register_hooks(self.netD_A)  # Re-register to reset features
        self.netD_A(fore_fake_B)  # Collect fake features
        self.features_fake_B = self.features.copy()
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.fore_real_B, fore_fake_B)

    def backward_D_B(self):
        fore_fake_A = self.fake_A_pool.query(self.fore_fake_A)
        self.register_hooks(self.netD_B)
        self.netD_B(self.fore_real_A)
        self.features_real_A = self.features.copy()
        self.register_hooks(self.netD_B)
        self.netD_B(fore_fake_A)
        self.features_fake_A = self.features.copy()
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.fore_real_A, fore_fake_A)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_perc = self.opt.lambda_perceptual  # Weight for perceptual loss

        # Identity loss
        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fore_fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fore_fake_A), True)

        # Cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # Background loss
        self.loss_background_A = self.criterionBackground(self.back_fake_B, self.back_real_A) * lambda_A * lambda_idt
        self.loss_background_B = self.criterionBackground(self.back_fake_A, self.back_real_B) * lambda_B * lambda_idt

        # Perceptual loss
        self.loss_perc_A = self.compute_perceptual_loss(self.features_real_A, self.features_fake_A) * lambda_perc
        self.loss_perc_B = self.compute_perceptual_loss(self.features_real_B, self.features_fake_B) * lambda_perc

        # Combined loss
        self.loss_G = (self.loss_G_A + self.loss_G_B + 
                       self.loss_cycle_A + self.loss_cycle_B + 
                       self.loss_idt_A + self.loss_idt_B + 
                       self.loss_background_A + self.loss_background_B + 
                       self.loss_perc_A + self.loss_perc_B)
        self.loss_G.backward()

    def optimize_parameters(self):
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

    # Placeholder for get_masking (assumed to be defined elsewhere)
    def get_masking(self, image, threshold):
        # Implement or import this function as needed
        pass
