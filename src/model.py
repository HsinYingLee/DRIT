import networks
import torch
from torch.autograd import Variable
import torch.nn as nn
import itertools

class DRIT(nn.Module):
  def __init__(self, opts):
    super(DRIT, self).__init__()

    self.nz = 8
    lr = 0.0001

    # discriminators
    self.disA = networks.Dis(opts.input_dim_a)
    self.disB = networks.Dis(opts.input_dim_b)
    self.disA2 = networks.Dis(opts.input_dim_a)
    self.disB2 = networks.Dis(opts.input_dim_b)
    self.disContent = networks.Dis_content()

    # encoders
    self.enc_c = networks.E_content(opts.input_dim_a, opts.input_dim_b)
    self.enc_a = networks.E_attr(opts.input_dim_a, opts.input_dim_b, self.nz,\
                                norm_layer=networks.get_norm_layer(layer_type='instance'),\
                                nl_layer=networks.get_non_linearity(layer_type='lrelu'))

    # generator
    self.gen = networks.G(opts.input_dim_a, opts.input_dim_b, nz=self.nz)

    # optimizers
    self.disA_opt = torch.optim.Adam(self.disA.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.disB_opt = torch.optim.Adam(self.disB.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.disA2_opt = torch.optim.Adam(self.disA2.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.disB2_opt = torch.optim.Adam(self.disB2.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.disContent_opt = torch.optim.Adam(self.disContent.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.enc_c_opt = torch.optim.Adam(self.enc_c.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.enc_a_opt = torch.optim.Adam(self.enc_a.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)

    # Setup the loss function for training
    self.criterionL1 = torch.nn.L1Loss()

  def initialize(self):
    self.disA.apply(networks.gaussian_weights_init)
    self.disB.apply(networks.gaussian_weights_init)
    self.disA2.apply(networks.gaussian_weights_init)
    self.disB2.apply(networks.gaussian_weights_init)
    self.disContent.apply(networks.gaussian_weights_init)
    self.gen.apply(networks.gaussian_weights_init)
    self.enc_c.apply(networks.gaussian_weights_init)
    self.enc_a.apply(networks.gaussian_weights_init)

  def set_scheduler(self, opts, last_ep=0):
    self.disA_sch = networks.get_scheduler(self.disA_opt, opts, last_ep)
    self.disB_sch = networks.get_scheduler(self.disB_opt, opts, last_ep)
    self.disA2_sch = networks.get_scheduler(self.disA2_opt, opts, last_ep)
    self.disB2_sch = networks.get_scheduler(self.disB2_opt, opts, last_ep)
    self.disContent_sch = networks.get_scheduler(self.disContent_opt, opts, last_ep)
    self.enc_c_sch = networks.get_scheduler(self.enc_c_opt, opts, last_ep)
    self.enc_a_sch = networks.get_scheduler(self.enc_a_opt, opts, last_ep)
    self.gen_sch = networks.get_scheduler(self.gen_opt, opts, last_ep)

  def setgpu(self, gpu):
    self.gpu = gpu
    self.disA.cuda(self.gpu)
    self.disB.cuda(self.gpu)
    self.disA2.cuda(self.gpu)
    self.disB2.cuda(self.gpu)
    self.disContent.cuda(self.gpu)
    self.enc_c.cuda(self.gpu)
    self.enc_a.cuda(self.gpu)
    self.gen.cuda(self.gpu)

  def get_z_random(self, batchSize, nz, random_type='gauss'):
    z = torch.cuda.FloatTensor(batchSize, nz)
    z.copy_(torch.randn(batchSize, nz))
    z = Variable(z)
    return z

  def get_z_random_test(self, batchSize, nz, random_type='gauss'):
    z = torch.cuda.FloatTensor(batchSize, nz)
    z.copy_(torch.randn(batchSize, nz))
    return z

  def test_forward_acas(self, image_ac, image_as):
    self.zs_ac, self.zs_as = self.enc_c.forward(image_ac, image_as)
    self.mu_as, self.logvar_as, self.mu_as_, self.logvar_as_ = self.enc_a.forward(image_as, image_as)
    std_a = self.logvar_as.mul(0.5).exp_()
    eps = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
    self.z_attr_as = eps.mul(std_a).add_(self.mu_as)
    image = self.gen.forward_a(self.zs_ac, self.z_attr_as)
    return image
  def test_forward_acbs(self, image_ac, image_bs):
    self.zs_ac, self.zs_as = self.enc_c.forward(image_ac, image_bs)
    self.mu_as, self.logvar_as, self.mu_bs, self.logvar_bs = self.enc_a.forward(image_bs, image_bs)
    std_b = self.logvar_bs.mul(0.5).exp_()
    eps = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
    self.z_attr_bs = eps.mul(std_b).add_(self.mu_bs)
    image = self.gen.forward_b(self.zs_ac, self.z_attr_bs)
    return image
  def test_forward_bcas(self, image_bc, image_as):
    self.zs_bc, self.zs_as = self.enc_c.forward(image_bc, image_as)
    self.mu_as, self.logvar_as, self.mu_as_, self.logvar_as_ = self.enc_a.forward(image_as, image_as)
    std_a = self.logvar_as.mul(0.5).exp_()
    eps = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
    self.z_attr_as = eps.mul(std_a).add_(self.mu_as)
    image = self.gen.forward_a(self.zs_bc, self.z_attr_as)
    return image
  def test_forward_bcbs(self, image_bc, image_bs):
    self.zs_bc, self.zs_bs = self.enc_c.forward(image_bc, image_bs)
    self.mu_bs_, self.logvar_bs_, self.mu_bs, self.logvar_bs = self.enc_a.forward(image_bs, image_bs)
    std_b = self.logvar_bs.mul(0.5).exp_()
    eps = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
    self.z_attr_bs = eps.mul(std_b).add_(self.mu_bs)
    image = self.gen.forward_b(self.zs_bc, self.z_attr_bs)
    return image

  def test_forward_a2b(self, image_a, num=5):
    self.zs_a, self.zs_b = self.enc_c.forward(image_a, image_a)
    # get random z
    self.z_random = self.get_z_random_test(num, self.nz, 'gauss')
    self.image_a2b = []
    for i in range(num):
      z = torch.cuda.FloatTensor(1,8)
      z.copy_(self.z_random[i])
      z = Variable(z)
      self.image_a2b.append(self.gen.forward_b(self.zs_a, z))
    return self.image_a2b

  def test_forward_b2a(self, image_b, num=5):
    self.zs_a, self.zs_b = self.enc_c.forward(image_b, image_b)
    # get random z
    self.z_random = self.get_z_random_test(num, self.nz, 'gauss')
    self.image_b2a = []
    for i in range(num):
      z = torch.cuda.FloatTensor(1,8)
      z.copy_(self.z_random[i])
      z = Variable(z)
      self.image_b2a.append(self.gen.forward_a(self.zs_b, z))
    return self.image_b2a

  def test_forward(self, image_a, image_b):
    self.zs_a, self.zs_b = self.enc_c.forward(image_a, image_b)
    self.mu_a, self.logvar_a, self.mu_b, self.logvar_b = self.enc_a.forward(image_a, image_b)
    std_a = self.logvar_a.mul(0.5).exp_()
    eps = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
    self.z_encoded_a = eps.mul(std_a).add_(self.mu_a)
    std_b = self.logvar_b.mul(0.5).exp_()
    eps = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
    self.z_encoded_b = eps.mul(std_b).add_(self.mu_b)
    # get random z
    self.z_random = self.get_z_random_test(5, self.nz, 'gauss')
    self.image_a2b = []
    self.image_b2a = []
    self.image_a2b.append(self.gen.forward_b(self.zs_a, self.z_encoded_b))
    self.image_b2a.append(self.gen.forward_a(self.zs_b, self.z_encoded_a))
    for i in range(5):
      z = torch.cuda.FloatTensor(std_a.size(0), std_a.size(1))
      z.copy_(self.z_random[i])
      z = Variable(z)
      self.image_a2b.append(self.gen.forward_b(self.zs_a, z))
      self.image_b2a.append(self.gen.forward_a(self.zs_b, z))
    return self.image_a2b, self.image_b2a

  def forward(self):
    # input images
    half_size = 1
    real_A = self.input_A
    real_B = self.input_B
    self.real_A_encoded = real_A[0:half_size]
    self.real_A_random = real_A[half_size:]
    self.real_B_encoded = real_B[0:half_size]
    self.real_B_random = real_B[half_size:]

    # get encoded z_c
    self.z_content_a, self.z_content_b = self.enc_c.forward(self.real_A_encoded, self.real_B_encoded)

    # get encoded z_a
    self.mu_a, self.logvar_a, self.mu_b, self.logvar_b = self.enc_a.forward(self.real_A_encoded, self.real_B_encoded)
    std_a = self.logvar_a.mul(0.5).exp_()
    eps_a = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
    self.z_attr_a = eps_a.mul(std_a).add_(self.mu_a)
    std_b = self.logvar_b.mul(0.5).exp_()
    eps_b = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
    self.z_attr_b = eps_b.mul(std_b).add_(self.mu_b)

    # get random z
    self.z_random = self.get_z_random(self.real_A_encoded.size(0), self.nz, 'gauss')

    # first cross translation
    input_content_forA = torch.cat((self.z_content_b, self.z_content_a, self.z_content_b),0)
    input_content_forB = torch.cat((self.z_content_a, self.z_content_b, self.z_content_a),0)
    input_attr_forA = torch.cat((self.z_attr_a, self.z_attr_a, self.z_random),0)
    input_attr_forB = torch.cat((self.z_attr_b, self.z_attr_b, self.z_random),0)
    output_fakeA = self.gen.forward_a(input_content_forA, input_attr_forA)
    output_fakeB = self.gen.forward_b(input_content_forB, input_attr_forB)
    self.fake_A_encoded, self.fake_AA_encoded, self.fake_A_random = torch.split(output_fakeA, self.z_content_a.size(0), dim=0)
    self.fake_B_encoded, self.fake_BB_encoded, self.fake_B_random = torch.split(output_fakeB, self.z_content_a.size(0), dim=0)

    # get encoded z_c and z_a
    self.z_content_recon_b, self.z_content_recon_a = self.enc_c.forward(self.fake_A_encoded, self.fake_B_encoded)
    self.mu_recon_a, self.logvar_recon_a, self.mu_recon_b, self.logvar_recon_b = self.enc_a.forward(self.fake_A_encoded, self.fake_B_encoded)
    std_a = self.logvar_recon_a.mul(0.5).exp_()
    eps_a = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
    self.z_attr_recon_a = eps_a.mul(std_a).add_(self.mu_recon_a)
    std_b = self.logvar_recon_b.mul(0.5).exp_()
    eps_b = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
    self.z_attr_recon_b = eps_b.mul(std_b).add_(self.mu_recon_b)

    # second cross translation
    self.fake_A_recon = self.gen.forward_a(self.z_content_recon_a, self.z_attr_recon_a)
    self.fake_B_recon = self.gen.forward_b(self.z_content_recon_b, self.z_attr_recon_b)

    # for display
    self.image_display = torch.cat((self.real_A_encoded[0:1].data.cpu(), self.fake_B_encoded[0:1].data.cpu(), \
                                    self.fake_B_random[0:1].data.cpu(), self.fake_AA_encoded[0:1].data.cpu(), self.fake_A_recon[0:1].data.cpu(), \
                                    self.real_B_encoded[0:1].data.cpu(), self.fake_A_encoded[0:1].data.cpu(), \
                                    self.fake_A_random[0:1].data.cpu(), self.fake_BB_encoded[0:1].data.cpu(), self.fake_B_recon[0:1].data.cpu()), dim=0)

    # for latent regression
    self.mu2_a, self.logvar2_a, self.mu2_b, self.logvar2_b = self.enc_a.forward(self.fake_A_random, self.fake_B_random)

  def forward_content(self):
    half_size = 1
    self.real_A_encoded = self.input_A[0:half_size]
    self.real_B_encoded = self.input_B[0:half_size]
    # get encoded z_c
    self.z_content_a, self.z_content_b = self.enc_c.forward(self.real_A_encoded, self.real_B_encoded)

  def update_D_content(self, image_a, image_b):
    self.input_A = image_a
    self.input_B = image_b
    self.forward_content()
    self.disContent_opt.zero_grad()
    loss_D_Content = self.backward_contentD(self.z_content_a, self.z_content_b)
    self.disContent_loss = loss_D_Content.data.cpu().numpy()[0]
    nn.utils.clip_grad_norm(self.disContent.parameters(), 5)
    self.disContent_opt.step()

  def update_D(self, image_a, image_b):
    self.input_A = image_a
    self.input_B = image_b
    self.forward()

    # update disA
    self.disA_opt.zero_grad()
    loss_D1_A = self.backward_D(self.disA, self.real_A_encoded, self.fake_A_encoded)
    self.disA_loss = loss_D1_A.data.cpu().numpy()[0]
    #nn.utils.clip_grad_norm(self.disA.parameters(), 5)
    self.disA_opt.step()

    # update disA2
    self.disA2_opt.zero_grad()
    loss_D2_A = self.backward_D(self.disA2, self.real_A_random, self.fake_A_random)
    self.disA2_loss = loss_D2_A.data.cpu().numpy()[0]
    #nn.utils.clip_grad_norm(self.disA2.parameters(), 5)
    self.disA2_opt.step()

    # update disB
    self.disB_opt.zero_grad()
    loss_D1_B = self.backward_D(self.disB, self.real_B_encoded, self.fake_B_encoded)
    self.disB_loss = loss_D1_B.data.cpu().numpy()[0]
    #nn.utils.clip_grad_norm(self.disB.parameters(), 5)
    self.disB_opt.step()

    # update disB2
    self.disB2_opt.zero_grad()
    loss_D2_B = self.backward_D(self.disB2, self.real_B_random, self.fake_B_random)
    self.disB2_loss = loss_D2_B.data.cpu().numpy()[0]
    #nn.utils.clip_grad_norm(self.disB2.parameters(), 5)
    self.disB2_opt.step()

    # update disContent
    self.disContent_opt.zero_grad()
    loss_D_Content = self.backward_contentD(self.z_content_a, self.z_content_b)
    self.disContent_loss = loss_D_Content.data.cpu().numpy()[0]
    nn.utils.clip_grad_norm(self.disContent.parameters(), 5)
    self.disContent_opt.step()

  def backward_D(self, netD, real, fake):
    pred_fake = netD.forward(fake.detach())
    pred_real = netD.forward(real)
    for it, (out_a, out_b) in enumerate(itertools.izip(pred_fake, pred_real)):
      out_fake = nn.functional.sigmoid(out_a)
      out_real = nn.functional.sigmoid(out_b)
      all1 = Variable(torch.ones((out_real.size(0))).cuda(self.gpu))
      all0 = Variable(torch.zeros((out_fake.size(0))).cuda(self.gpu))
      ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
      ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
    loss_D = ad_true_loss + ad_fake_loss
    loss_D.backward()
    return loss_D

  def backward_contentD(self, imageA, imageB):
    pred_fake = self.disContent.forward(imageA.detach())
    pred_real = self.disContent.forward(imageB.detach())
    for it, (out_a, out_b) in enumerate(itertools.izip(pred_fake, pred_real)):
      out_fake = nn.functional.sigmoid(out_a)
      out_real = nn.functional.sigmoid(out_b)
      all1 = Variable(torch.ones((out_real.size(0))).cuda(self.gpu))
      all0 = Variable(torch.zeros((out_fake.size(0))).cuda(self.gpu))
      ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
      ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
    loss_D = ad_true_loss + ad_fake_loss
    loss_D.backward()
    return loss_D

  def update_EG(self):
    # update G and E
    self.enc_c_opt.zero_grad()
    self.enc_a_opt.zero_grad()
    self.gen_opt.zero_grad()
    self.backward_EG()
    '''nn.utils.clip_grad_norm(self.enc_c.parameters(), 5)
    nn.utils.clip_grad_norm(self.enc_a.parameters(), 5)
    nn.utils.clip_grad_norm(self.gen.parameters(), 5)'''
    self.enc_c_opt.step()
    self.enc_a_opt.step()
    self.gen_opt.step()

    self.enc_c_opt.zero_grad()
    self.gen_opt.zero_grad()
    self.backward_G_alone()
    '''nn.utils.clip_grad_norm(self.enc_c.parameters(), 5)
    nn.utils.clip_grad_norm(self.gen.parameters(), 5)'''
    self.enc_c_opt.step()
    self.gen_opt.step()

  def backward_EG(self):
    # content Ladv for generator
    loss_G_GAN_Acontent = self.backward_G_GAN_content(self.z_content_a)
    loss_G_GAN_Bcontent = self.backward_G_GAN_content(self.z_content_b)

    # Ladv for generator
    loss_G_GAN_A = self.backward_G_GAN(self.fake_A_encoded, self.disA)
    loss_G_GAN_B = self.backward_G_GAN(self.fake_B_encoded, self.disB)

    # KL loss - z_a
    kl_element_a = self.mu_a.pow(2).add_(self.logvar_a.exp()).mul_(-1).add_(1).add_(self.logvar_a)
    loss_kl_za = torch.sum(kl_element_a).mul_(-0.5) * 0.01
    kl_element_b = self.mu_b.pow(2).add_(self.logvar_b.exp()).mul_(-1).add_(1).add_(self.logvar_b)
    loss_kl_zb = torch.sum(kl_element_b).mul_(-0.5) * 0.01
    kl_element_recon_a = self.mu_recon_a.pow(2).add_(self.logvar_recon_a.exp()).mul_(-1).add_(1).add_(self.logvar_recon_a)
    loss_kl_recon_za = torch.sum(kl_element_recon_a).mul_(-0.5) * 0.01
    kl_element_recon_b = self.mu_recon_b.pow(2).add_(self.logvar_recon_b.exp()).mul_(-1).add_(1).add_(self.logvar_recon_b)
    loss_kl_recon_zb = torch.sum(kl_element_recon_b).mul_(-0.5) * 0.01

    # KL loss - z_c
    loss_kl_zs_a = self._compute_kl(self.z_content_a) * 0.01
    loss_kl_zs_b = self._compute_kl(self.z_content_b) * 0.01
    loss_kl_zs_recon_a = self._compute_kl(self.z_content_recon_a) * 0.01
    loss_kl_zs_recon_b = self._compute_kl(self.z_content_recon_b) * 0.01

    # cross cycle consistency loss
    loss_G_L1_A = self.criterionL1(self.fake_A_recon, self.real_A_encoded) * 10
    loss_G_L1_B = self.criterionL1(self.fake_B_recon, self.real_B_encoded) * 10
    loss_G_L1_AA = self.criterionL1(self.fake_AA_encoded, self.real_A_encoded) * 10
    loss_G_L1_BB = self.criterionL1(self.fake_BB_encoded, self.real_B_encoded) * 10

    loss_G = loss_G_GAN_A + loss_G_GAN_B + \
             loss_G_GAN_Acontent + loss_G_GAN_Bcontent + \
             loss_kl_za + loss_kl_zb + loss_kl_recon_za + loss_kl_recon_zb + \
             loss_kl_zs_a + loss_kl_zs_b + loss_kl_zs_recon_a + loss_kl_zs_recon_b + \
             loss_G_L1_AA + loss_G_L1_BB +\
             loss_G_L1_A + loss_G_L1_B

    loss_G.backward(retain_graph=True)

    self.gan_loss_a = loss_G_GAN_A.data.cpu().numpy()[0]
    self.gan_loss_b = loss_G_GAN_B.data.cpu().numpy()[0]
    self.gan_loss_acontent = loss_G_GAN_Acontent.data.cpu().numpy()[0]
    self.gan_loss_bcontent = loss_G_GAN_Bcontent.data.cpu().numpy()[0]
    self.kl_za_loss = loss_kl_za.data.cpu().numpy()[0]
    self.kl_zb_loss = loss_kl_zb.data.cpu().numpy()[0]
    self.kl_zs_a_loss = loss_kl_zs_a.data.cpu().numpy()[0]
    self.kl_zs_b_loss = loss_kl_zs_b.data.cpu().numpy()[0]
    self.l1_recon_A_loss = loss_G_L1_A.data.cpu().numpy()[0]
    self.l1_recon_B_loss = loss_G_L1_B.data.cpu().numpy()[0]
    self.l1_recon_AA_loss = loss_G_L1_AA.data.cpu().numpy()[0]
    self.l1_recon_BB_loss = loss_G_L1_BB.data.cpu().numpy()[0]
    self.G_loss = loss_G.data.cpu().numpy()[0]

  def backward_G_GAN_content(self, data):
    outs = self.disContent.forward(data)
    for out in outs:
      outputs_fake = nn.functional.sigmoid(out)
      all_half = Variable(0.5*torch.ones((outputs_fake.size(0))).cuda(self.gpu))
      ad_loss = nn.functional.binary_cross_entropy(outputs_fake, all_half)
    return ad_loss

  def backward_G_GAN(self, fake, netD=None):
    outs_fake = netD.forward(fake)
    for out_a in outs_fake:
      outputs_fake = nn.functional.sigmoid(out_a)
      all_ones = Variable(torch.ones((outputs_fake.size(0))).cuda(self.gpu))
      ad_loss_a = nn.functional.binary_cross_entropy(outputs_fake, all_ones)
    return ad_loss_a

  def backward_G_alone(self):
    # Ladv for generator
    loss_G_GAN2_A = self.backward_G_GAN(self.fake_A_random, self.disA2)
    loss_G_GAN2_B = self.backward_G_GAN(self.fake_B_random, self.disB2)

    # latent regression loss
    loss_z_L1_a = torch.mean(torch.abs(self.mu2_a - self.z_random)) * 10
    loss_z_L1_b = torch.mean(torch.abs(self.mu2_b - self.z_random)) * 10

    loss_z_L1 = loss_z_L1_a + loss_z_L1_b + loss_G_GAN2_A + loss_G_GAN2_B
    loss_z_L1.backward()
    self.l1_recon_z_loss_a = loss_z_L1_a.data.cpu().numpy()[0]
    self.l1_recon_z_loss_b = loss_z_L1_b.data.cpu().numpy()[0]
    self.gan2_loss_a = loss_G_GAN2_A.data.cpu().numpy()[0]
    self.gan2_loss_b = loss_G_GAN2_B.data.cpu().numpy()[0]

  def update_lr(self):
    self.disA_sch.step()
    self.disB_sch.step()
    self.disA2_sch.step()
    self.disB2_sch.step()
    self.disContent_sch.step()
    self.enc_c_sch.step()
    self.enc_a_sch.step()
    self.gen_sch.step()

  def _compute_kl(self, mu):
    mu_2 = torch.pow(mu, 2)
    encoding_loss = torch.mean(mu_2)
    return encoding_loss

  def resume(self, model_dir):
    checkpoint = torch.load(model_dir)
    # weight
    self.disA.load_state_dict(checkpoint['disA'])
    self.disA2.load_state_dict(checkpoint['disA2'])
    self.disB.load_state_dict(checkpoint['disB'])
    self.disB2.load_state_dict(checkpoint['disB2'])
    self.disContent.load_state_dict(checkpoint['disContent'])
    self.enc_c.load_state_dict(checkpoint['enc_c'])
    self.enc_a.load_state_dict(checkpoint['enc_a'])
    self.gen.load_state_dict(checkpoint['gen'])
    # optimizer
    self.disA_opt.load_state_dict(checkpoint['disA_opt'])
    self.disA2_opt.load_state_dict(checkpoint['disA2_opt'])
    self.disB_opt.load_state_dict(checkpoint['disB_opt'])
    self.disB2_opt.load_state_dict(checkpoint['disB2_opt'])
    self.disContent_opt.load_state_dict(checkpoint['disContent_opt'])
    self.enc_c_opt.load_state_dict(checkpoint['enc_c_opt'])
    self.enc_a_opt.load_state_dict(checkpoint['enc_a_opt'])
    self.gen_opt.load_state_dict(checkpoint['gen_opt'])
    return checkpoint['ep']

  def save(self, filename, ep):
    state = {
             'disA': self.disA.state_dict(),
             'disA2': self.disA2.state_dict(),
             'disB': self.disB.state_dict(),
             'disB2': self.disB2.state_dict(),
             'disContent': self.disContent.state_dict(),
             'enc_c': self.enc_c.state_dict(),
             'enc_a': self.enc_a.state_dict(),
             'gen': self.gen.state_dict(),
             'disA_opt': self.disA_opt.state_dict(),
             'disA2_opt': self.disA2_opt.state_dict(),
             'disB_opt': self.disB_opt.state_dict(),
             'disB2_opt': self.disB2_opt.state_dict(),
             'disContent_opt': self.disContent_opt.state_dict(),
             'enc_c_opt': self.enc_c_opt.state_dict(),
             'enc_a_opt': self.enc_a_opt.state_dict(),
             'gen_opt': self.gen_opt.state_dict(),
             'ep': ep
              }
    torch.save(state, filename)
    return

  def assemble_outputs3(self):
    images_a = self.normalize_image(self.real_A_encoded)
    images_b = self.normalize_image(self.real_B_encoded)
    images_a1 = self.normalize_image(self.fake_A_encoded)
    images_a2 = self.normalize_image(self.fake_A_random)
    images_a3 = self.normalize_image(self.fake_A_recon)
    images_a4 = self.normalize_image(self.fake_AA_encoded)
    images_b1 = self.normalize_image(self.fake_B_encoded)
    images_b2 = self.normalize_image(self.fake_B_random)
    images_b3 = self.normalize_image(self.fake_B_recon)
    images_b4 = self.normalize_image(self.fake_BB_encoded)
    row1 = torch.cat((images_a[0:1, ::], images_b1[0:1, ::], images_b2[0:1, ::], images_a4[0:1, ::], images_a3[0:1, ::]),3)
    row2 = torch.cat((images_b[0:1, ::], images_a1[0:1, ::], images_a2[0:1, ::], images_b4[0:1, ::], images_b3[0:1, ::]),3)
    return torch.cat((row1,row2),2)

  def normalize_image(self, x):
    return x[:,0:3,:,:]


