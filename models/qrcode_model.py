import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys
import random
import torch
import torch.nn.functional as F


class CQRCodeGANModel(BaseModel):
    def name(self):
        return 'CQRCodeGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        # self.input_A = self.Tensor(nb, opt.input_nc, size, size)   # input image channels
        self.real_A = self.Tensor(nb, opt.input_nc, size, size)   # input image channels
        # self.input_B = self.Tensor(nb, opt.output_nc, size, size)  #then crop to this size
        self.real_B = self.Tensor(nb, opt.output_nc, size, size)  #then crop to this size
        
        self.input_img = self.Tensor(nb, opt.input_nc, size, size)
        self.qrcode_position = self.Tensor(nb, 4)  # A的二维码的位置
        self.input_A_gray = self.Tensor(nb, 1, size, size)  #attention
        self.real_patch_B_list = []
        self.fake_patch_B_list = []
        self.real_patch_A_list = []
        self.fake_patch_A_list = []
        
        if opt.vgg > 0:
            self.vgg_loss = networks.PerceptualLoss(opt)
            self.vgg_loss.cuda()
            self.vgg = networks.load_vgg16(opt.vgg_pretrained_path,opt.gpu_ids)
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        skip = True if opt.skip > 0 else False
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, skip=skip, opt=opt)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, skip=False, opt=opt)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
            if opt.patch_D:
                self.netD_A_patch = networks.define_D(opt.input_nc, opt.ndf,
                                                      opt.which_model_netD, opt.n_layers_patchD, 
                                                      opt.norm, use_sigmoid, self.gpu_ids, True)
            else:
                self.netD_A_patch = None

            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
            if opt.patch_D:
                self.netD_B_patch = networks.define_D(opt.input_nc, opt.ndf,
                                                      opt.which_model_netD, opt.n_layers_patchD,
                                                      opt.norm, use_sigmoid, self.gpu_ids, True)
            else:
                self.netD_B_patch = None
            
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)  #当query的时候就会发生更新
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.fake_patch_A_pool = ImagePool(opt.pool_size)
            self.fake_patch_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            if opt.use_wgan:
                self.criterionGAN = networks.DiscLossWGANGP()
            else:
                self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if self.netD_A_patch is not None:
                self.optimizer_D_A_patch = torch.optim.Adam(self.netD_A_patch.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if self.netD_B_patch is not None:
                self.optimizer_D_B_patch = torch.optim.Adam(self.netD_B_patch.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_B)

        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
            if self.netD_A_patch is not None:
                networks.print_network(self.netD_A_patch)
            if self.netD_B_patch is not None:
                networks.print_network(self.netD_B_patch)
        if opt.isTrain:
            self.netG_A.train()
            self.netG_B.train()
        else:
            self.netG_A.eval()
            self.netG_B.eval()
        print('-----------------------------------------------')

    def set_input(self, input):
        # AtoB = self.opt.which_direction == 'AtoB'
        # input_A = input['A' if AtoB else 'B']
        # input_B = input['B' if AtoB else 'A']
        # self.input_A.resize_(input_A.size()).copy_(input_A)
        # self.input_B.resize_(input_B.size()).copy_(input_B)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']


        AtoB = self.opt.which_direction == 'AtoB'
        # _A = input['A' if AtoB else 'B']
        # input_B = input['B' if AtoB else 'A']
        input_img = input['input_img']  # input_img就是A_img
        input_A_gray = input['A_gray']
        # self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_A_gray.resize_(input_A_gray.size()).copy_(input_A_gray)
        self.qrcode_position.resize_(input['qrcode_position'].size()).copy_(input['qrcode_position'])
        # self.attention_img.resize_(input['attention_img'].size()).copy_(input['attention_img'])
        # self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_img.resize_(input_img.size()).copy_(input_img)
        real_A = input['A' if AtoB else 'B']
        self.real_A.resize_(real_A.size()).copy_(real_A)
        real_B = input['B' if AtoB else 'A']
        self.real_B.resize_(real_B.size()).copy_(real_B)
        
        
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        # self.real_A = Variable(self.input_A)
        # self.real_B = Variable(self.input_B)
        
        
        # if self.opt.noise > 0:
        #     self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise/255.))
        #     self.real_A = self.real_A + self.noise
        # if self.opt.input_linear:  # False
        #     self.real_A = (self.real_A - torch.min(self.real_A))/(torch.max(self.real_A) - torch.min(self.real_A))
        if self.opt.skip == 1:  # True
            self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A)
        else:
            self.fake_B = self.netG_A.forward(self.real_A)
        
        self.rec_A = self.netG_B.forward(self.fake_B)
        self.fake_A = self.netG_B.forward(self.real_B)
        if self.opt.skip == 1:
            self.rec_B , self.latent_fake_A = self.netG_A.forward(self.fake_A)
        else:
            self.rec_B = self.netG_A.forward(self.fake_A)
        if self.isTrain and self.opt.patch_D:
            # Calculate region widths and heights
            region_w = self.qrcode_position[:, 2] - self.qrcode_position[:, 0]
            region_h = self.qrcode_position[:, 3] - self.qrcode_position[:, 1]
            
            # Calculate maximum offsets
            w_offset_max = torch.clamp(region_w - self.opt.patchSize, min=0)
            h_offset_max = torch.clamp(region_h - self.opt.patchSize, min=0)
            
            # for i in range(self.qrcode_position.size(0)):
            #     assert w_offset_max[i].item() >= 0
            #     assert h_offset_max[i].item() >= 0
            # Generate random offsets
            w_offsets = self.qrcode_position[:, 0] + torch.tensor([torch.randint(0, int((w_offset_max[i] + 1).item()),[1]) for i in range(self.qrcode_position.size(0))]).to(self.qrcode_position.device)  #生成batch_size大小的随机偏移
            h_offsets = self.qrcode_position[:, 1] + torch.tensor([torch.randint(0, int((h_offset_max[i] + 1).item()),[1]) for i in range(self.qrcode_position.size(0))]).to(self.qrcode_position.device)
            
            # Clamp offsets to ensure they are within image boundaries
            w_offsets = torch.clamp(w_offsets, min=0, max=self.real_B.size(3) - self.opt.patchSize)  #设置patch的x1范围，防止出界
            h_offsets = torch.clamp(h_offsets, min=0, max=self.real_B.size(2) - self.opt.patchSize)
            
            # Extract patches using advanced indexing
            # batch_indices = torch.arange(self.real_B.size(0)).to(self.real_B.device)
            
            self.real_patch_B = []
            self.fake_patch_B = []
            self.real_patch_A = []
            self.fake_patch_A = []

            for idx in range(self.qrcode_position.size(0)):
                h = int(h_offsets[idx].item())
                w = int(w_offsets[idx].item())
                
                real_patch_B = self.real_B[idx, :, h:h + self.opt.patchSize, w:w + self.opt.patchSize]
                fake_patch_B = self.fake_B[idx, :, h:h + self.opt.patchSize, w:w + self.opt.patchSize]
                real_patch_A = self.real_A[idx, :, h:h + self.opt.patchSize, w:w + self.opt.patchSize]
                fake_patch_A = self.fake_A[idx, :, h:h + self.opt.patchSize, w:w + self.opt.patchSize]
                
                self.real_patch_B.append(real_patch_B)
                self.fake_patch_B.append(fake_patch_B)
                self.real_patch_A.append(real_patch_A)
                self.fake_patch_A.append(fake_patch_A)

            self.real_patch_B = torch.stack(self.real_patch_B)
            self.fake_patch_B = torch.stack(self.fake_patch_B)
            self.real_patch_A = torch.stack(self.real_patch_A)
            self.fake_patch_A = torch.stack(self.fake_patch_A)
            
        # 如果 patch_D_3 大于 0，则循环截取多个 patch
        # 如果 patch_D_3 大于 0，则循环截取多个 patch
            if getattr(self.opt, 'patch_D_3', 0) > 0:
                self.real_patch_B_list = []
                self.fake_patch_B_list = []
                self.real_patch_A_list = []
                self.fake_patch_A_list = []

                for i in range(self.opt.patch_D_3):
                    # 重新计算偏移量，类似 patch_D 的逻辑
                    w_offsets = self.qrcode_position[:, 0] + torch.tensor([torch.randint(0, int((w_offset_max[i] + 1).item()),[1]) for i in range(self.qrcode_position.size(0))]).to(self.qrcode_position.device)

                    h_offsets = self.qrcode_position[:, 1] + torch.tensor([torch.randint(0, int((h_offset_max[i] + 1).item()),[1]) for i in range(self.qrcode_position.size(0))]).to(self.qrcode_position.device)
                    
                    # Clamp 偏移量以确保不会越界
                    w_offsets = torch.clamp(w_offsets, min=0, max=self.real_B.size(3) - self.opt.patchSize)
                    h_offsets = torch.clamp(h_offsets, min=0, max=self.real_B.size(2) - self.opt.patchSize)

                    # 提取 patch
                    real_patch_B = []
                    fake_patch_B = []
                    real_patch_A = []
                    fake_patch_A = []

                    for idx in range(self.qrcode_position.size(0)):
                        h = int(h_offsets[idx].item())
                        w = int(w_offsets[idx].item())

                        real_patch_B.append(self.real_B[idx, :, h:h + self.opt.patchSize, w:w + self.opt.patchSize])
                        fake_patch_B.append(self.fake_B[idx, :, h:h + self.opt.patchSize, w:w + self.opt.patchSize])
                        real_patch_A.append(self.real_A[idx, :, h:h + self.opt.patchSize, w:w + self.opt.patchSize])
                        fake_patch_A.append(self.fake_A[idx, :, h:h + self.opt.patchSize, w:w + self.opt.patchSize])

                    self.real_patch_B_list.append(torch.stack(real_patch_B))
                    self.fake_patch_B_list.append(torch.stack(fake_patch_B))
                    self.real_patch_A_list.append(torch.stack(real_patch_A))
                    self.fake_patch_A_list.append(torch.stack(fake_patch_A))



    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        # print(np.transpose(self.real_A.data[0].cpu().float().numpy(),(1,2,0))[:2][:2][:])
        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A)
        else:
            self.fake_B = self.netG_A.forward(self.real_A)
        self.rec_A = self.netG_B.forward(self.fake_B)

        self.real_B = Variable(self.input_B, volatile=True)
        self.fake_A = self.netG_B.forward(self.real_B)
        if self.opt.skip == 1:
            self.rec_B, self.latent_fake_A = self.netG_A.forward(self.fake_A)
        else:
            self.rec_B = self.netG_A.forward(self.fake_A)

    def predict(self):
        # self.real_A = Variable(self.input_A, volatile=True)
        # print(np.transpose(self.real_A.data[0].cpu().float().numpy(),(1,2,0))[:2][:2][:])
        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A)
        else:
            self.fake_B = self.netG_A.forward(self.real_A)
        self.rec_A = self.netG_B.forward(self.fake_B)

        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        rec_A = util.tensor2im(self.rec_A.data)
        if self.opt.skip == 1:
            latent_real_A = util.tensor2im(self.latent_real_A.data)
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ("latent_real_A", latent_real_A), ("rec_A", rec_A)])
        else:
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ("rec_A", rec_A)])

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    # def backward_D_basic(self, netD, real, fake):
    #     # Real
    #     pred_real = netD.forward(real)
    #     if self.opt.use_wgan:
    #         loss_D_real = pred_real.mean()
    #     else:
    #         loss_D_real = self.criterionGAN(pred_real, True)
    #     # Fake
    #     pred_fake = netD.forward(fake.detach())
    #     if self.opt.use_wgan:
    #         loss_D_fake = pred_fake.mean()
    #     else:
    #         loss_D_fake = self.criterionGAN(pred_fake, False)
    #     # Combined loss
    #     if self.opt.use_wgan:
    #         loss_D = loss_D_fake - loss_D_real + self.criterionGAN.calc_gradient_penalty(netD, real.data, fake.data)
    #     else:
    #         loss_D = (loss_D_real + loss_D_fake) * 0.5
    #     # backward
    #     loss_D.backward()
    #     return loss_D



    def backward_D_basic(self, netD, real, fake, use_ragan):  #增加ragan
        # Real
        pred_real = netD.forward(real)
        pred_fake = netD.forward(fake.detach())
        if self.opt.use_wgan:   #0
            loss_D_real = pred_real.mean()
            loss_D_fake = pred_fake.mean()
            loss_D = loss_D_fake - loss_D_real + self.criterionGAN.calc_gradient_penalty(netD, 
                                                real.data, fake.data)
        elif self.opt.use_ragan and use_ragan:       #true ragan
            loss_D = (self.criterionGAN(pred_real - torch.mean(pred_fake), True) +
                                      self.criterionGAN(pred_fake - torch.mean(pred_real), False)) / 2
        else:
            loss_D_real = self.criterionGAN(pred_real, True)  #标准GAN
            loss_D_fake = self.criterionGAN(pred_fake, False)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        # loss_D.backward()
        return loss_D
    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B,True) #使用ragan
        self.loss_D_A.backward()

    def backward_D_A_patch(self):
        fake_patch_B = self.fake_patch_B_pool.query(self.fake_patch_B)
        self.loss_D_A_patch = self.backward_D_basic(self.netD_A_patch, self.real_patch_B, fake_patch_B, False)
        if getattr(self.opt, 'patch_D_3', 0) > 0:
            total_loss = self.loss_D_A_patch
            for i in range(self.opt.patch_D_3):
                fake_patch_B = self.fake_patch_B_pool.query(self.fake_patch_B_list[i])
                total_loss += self.backward_D_basic(self.netD_A_patch, self.real_patch_B_list[i], fake_patch_B, False)
            self.loss_D_A_patch = total_loss / float(self.opt.patch_D_3 + 1)
        self.loss_D_A_patch.backward()

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A,True)
        self.loss_D_B.backward()

    def backward_D_B_patch(self):
        fake_patch_A = self.fake_patch_A_pool.query(self.fake_patch_A)
        self.loss_D_B_patch = self.backward_D_basic(self.netD_B_patch, self.real_patch_A, fake_patch_A, False)  #这里不
        if getattr(self.opt, 'patch_D_3', 0) > 0:
            total_loss = self.loss_D_B_patch
            for i in range(self.opt.patch_D_3):
                fake_patch_A = self.fake_patch_A_pool.query(self.fake_patch_A_list[i])
                total_loss += self.backward_D_basic(self.netD_B_patch, self.real_patch_A_list[i], fake_patch_A, False)
            self.loss_D_B_patch = total_loss / float(self.opt.patch_D_3 + 1)
        self.loss_D_B_patch.backward()

    def backward_G(self, epoch):  #在这一步不使用ragan
        lambda_idt = self.opt.identity   #可能可以不需要这个
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # GAN loss
        # D_A(G_A(A))
        # if self.opt.skip == 1:
        #     self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A)
        # else:
        #     self.fake_B = self.netG_A.forward(self.real_A)
         # = self.latent_real_A + self.opt.skip * self.real_A
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            if self.opt.skip == 1:             
                self.idt_A,self.latent_real_B = self.netG_A.forward(self.real_B)
            else:
                self.idt_A = self.netG_A.forward(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B.forward(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
            
            
            
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # if self.opt.l1 > 0:
        #     self.L1_AB = self.criterionL1(self.fake_B, self.real_B) * self.opt.l1
        # else:
        #     self.L1_AB = 0
        # Identity loss
        # D_B(G_B(B))
    
        # if self.opt.l1 > 0:
        #     self.L1_BA = self.criterionL1(self.fake_A, self.real_A) * self.opt.l1
        # else:
        #     self.L1_BA = 0

        # Forward cycle loss
        
        if lambda_A > 0:
            self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        else:
            self.loss_cycle_A = 0
        # Backward cycle loss
        
         # = self.latent_fake_A + self.opt.skip * self.fake_A
        if lambda_B > 0:
            self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        else:
            self.loss_cycle_B = 0
        self.loss_vgg_a = self.vgg_loss.compute_vgg_loss(self.vgg, self.fake_A, self.real_B) * self.opt.vgg if self.opt.vgg > 0 else 0
        self.loss_vgg_b = self.vgg_loss.compute_vgg_loss(self.vgg, self.fake_B, self.real_A) * self.opt.vgg if self.opt.vgg > 0 else 0
        if epoch <= 3:
            self.loss_vgg_a = 0
            self.loss_vgg_b = 0
        # PatchGAN
        if self.opt.patch_D:
            pred_fake_patch_B = self.netD_A_patch.forward(self.fake_patch_B)
            self.loss_G_A_patch = self.criterionGAN(pred_fake_patch_B, True)
            if getattr(self.opt, 'patch_D_3', 0) > 0:
                for i in range(self.opt.patch_D_3):
                    pred_fake_patch_B2 = self.netD_A_patch.forward(self.fake_patch_B_list[i])
                    self.loss_G_A_patch += self.criterionGAN(pred_fake_patch_B2, True)
                self.loss_G_A_patch = self.loss_G_A_patch / float(self.opt.patch_D_3 + 1)

            pred_fake_patch_A = self.netD_B_patch.forward(self.fake_patch_A)
            self.loss_G_B_patch = self.criterionGAN(pred_fake_patch_A, True)
            if getattr(self.opt, 'patch_D_3', 0) > 0:
                for i in range(self.opt.patch_D_3):
                    pred_fake_patch_A2 = self.netD_B_patch.forward(self.fake_patch_A_list[i])
                    self.loss_G_B_patch += self.criterionGAN(pred_fake_patch_A2, True)
                self.loss_G_B_patch = self.loss_G_B_patch / float(self.opt.patch_D_3 + 1)
        else:
            self.loss_G_A_patch = 0
            self.loss_G_B_patch = 0

        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B\
                      + self.loss_G_A_patch + self.loss_G_B_patch + self.loss_vgg_a + self.loss_vgg_b
        # combined loss
        self.loss_G.backward()

    def optimize_parameters(self, epoch):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G(epoch)
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()

        if self.opt.patch_D:
            self.optimizer_D_A_patch.zero_grad()
            self.backward_D_A_patch()
            self.optimizer_D_A_patch.step()

        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()
        if self.opt.patch_D:
            self.optimizer_D_B_patch.zero_grad()
            self.backward_D_B_patch()
            self.optimizer_D_B_patch.step()

    def get_current_errors(self, epoch):
        D_A = self.loss_D_A.item()
        G_A = self.loss_G_A.item()
        D_A_patch = self.loss_D_A_patch.item() if self.loss_D_A_patch is not None else 0
        Cyc_A = self.loss_cycle_A.item()
        D_B = self.loss_D_B.item()
        G_B = self.loss_G_B.item()
        D_B_patch = self.loss_D_B_patch.item() if self.loss_D_B_patch is not None else 0
        Cyc_B = self.loss_cycle_B.item()
        vgg = (self.loss_vgg_a.item() + self.loss_vgg_b.item()) / self.opt.vgg if epoch > 10 and self.opt.vgg > 0 else 0
        if self.opt.lambda_A > 0.0:
            return OrderedDict([('D_A', D_A), ('D_A_patch', D_A_patch), ('G_A', G_A), ('Cyc_A', Cyc_A),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B), ("vgg", vgg)])
        else:
            return OrderedDict([('D_A', D_A), ('D_B_patch', D_B_patch), ('G_A', G_A), 
                                ('D_B', D_B), ('G_B', G_B), ("vgg", vgg)])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        if self.opt.skip > 0:
            latent_real_A = util.tensor2im(self.latent_real_A.data)
        
        real_B = util.tensor2im(self.real_B.data)
        fake_A = util.tensor2im(self.fake_A.data)
        
        if self.opt.lambda_A > 0.0:
            rec_A = util.tensor2im(self.rec_A.data)
            rec_B = util.tensor2im(self.rec_B.data)
            if self.opt.skip > 0:
                latent_fake_A = util.tensor2im(self.latent_fake_A.data)
                visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A), ('rec_A', rec_A), 
                                       ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B), ('latent_fake_A', latent_fake_A)])
            else:
                visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A), 
                                       ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])
        else:
            if self.opt.skip > 0:
                visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A), 
                                       ('real_B', real_B), ('fake_A', fake_A)])
            else:
                visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B),
                                       ('real_B', real_B), ('fake_A', fake_A)])
        
        if self.opt.patch_D:
            real_patch_A = util.tensor2im(self.real_patch_A.data)
            fake_patch_A = util.tensor2im(self.fake_patch_A.data)
            real_patch_B = util.tensor2im(self.real_patch_B.data)
            fake_patch_B = util.tensor2im(self.fake_patch_B.data)
            visuals.update({'real_patch_A': real_patch_A, 'fake_patch_A': fake_patch_A,
                            'real_patch_B': real_patch_B, 'fake_patch_B': fake_patch_B})
        
        return visuals

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
        if self.netD_A_patch is not None:
            self.save_network(self.netD_A_patch, 'D_A_patch', label, self.gpu_ids)
        if self.netD_B_patch is not None:
            self.save_network(self.netD_B_patch, 'D_B_patch', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_B.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.netD_A_patch is not None:
            for param_group in self.optimizer_D_A_patch.param_groups:
                param_group['lr'] = lr
        if self.netD_B_patch is not None:
            for param_group in self.optimizer_D_B_patch.param_groups:
                param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
