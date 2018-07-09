import  torch
import torch.nn as nn
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F

# -------- spectral normalization ----------
class SpectralNorm(object):
  def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
    self.name = name
    self.dim = dim
    if n_power_iterations <= 0:
      raise ValueError('Expected n_power_iterations to be positive, but '
                       'got n_power_iterations={}'.format(n_power_iterations))
    self.n_power_iterations = n_power_iterations
    self.eps = eps
  def compute_weight(self, module):
    weight = getattr(module, self.name + '_orig')
    u = getattr(module, self.name + '_u')
    weight_mat = weight
    if self.dim != 0:
      # permute dim to front
      weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
    height = weight_mat.size(0)
    weight_mat = weight_mat.reshape(height, -1)
    with torch.no_grad():
      for _ in range(self.n_power_iterations):
        v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
        u = F.normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)
    sigma = torch.dot(u, torch.matmul(weight_mat, v))
    weight = weight / sigma
    return weight, u
  def remove(self, module):
    weight = getattr(module, self.name)
    delattr(module, self.name)
    delattr(module, self.name + '_u')
    delattr(module, self.name + '_orig')
    module.register_parameter(self.name, torch.nn.Parameter(weight))
  def __call__(self, module, inputs):
    if module.training:
      weight, u = self.compute_weight(module)
      setattr(module, self.name, weight)
      setattr(module, self.name + '_u', u)
    else:
      r_g = getattr(module, self.name + '_orig').requires_grad
      getattr(module, self.name).detach_().requires_grad_(r_g)

  @staticmethod
  def apply(module, name, n_power_iterations, dim, eps):
    fn = SpectralNorm(name, n_power_iterations, dim, eps)
    weight = module._parameters[name]
    height = weight.size(dim)
    u = F.normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
    delattr(module, fn.name)
    module.register_parameter(fn.name + "_orig", weight)
    module.register_buffer(fn.name, weight.data)
    module.register_buffer(fn.name + "_u", u)
    module.register_forward_pre_hook(fn)
    return fn

def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
  if dim is None:
    if isinstance(module, (torch.nn.ConvTranspose1d,
                           torch.nn.ConvTranspose2d,
                           torch.nn.ConvTranspose3d)):
      dim = 1
    else:
      dim = 0
  SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
  return module

def remove_spectral_norm(module, name='weight'):
  for k, hook in module._forward_pre_hooks.items():
    if isinstance(hook, SpectralNorm) and hook.name == name:
      hook.remove(module)
      del module._forward_pre_hooks[k]
      return module
  raise ValueError("spectral_norm of '{}' not found in {}".format(name, module))

# --------------- functions ------------------
def get_scheduler(optimizer, opts, cur_ep=-1):
  if opts.lr_policy == 'lambda':
    def lambda_rule(ep):
      lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / float(opts.n_ep - opts.n_ep_decay + 1)
      return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
  elif opts.lr_policy == 'step':
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
  else:
    return NotImplementedError('no such learn rate policy')
  return scheduler

def meanpoolConv(inplanes, outplanes):
  sequence = []
  sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
  sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)]
  return nn.Sequential(*sequence)

def convMeanpool(inplanes, outplanes):
  sequence = []
  sequence += [conv3x3(inplanes, outplanes)]
  sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
  return nn.Sequential(*sequence)

def get_norm_layer(layer_type='instance'):
  if layer_type == 'batch':
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
  elif layer_type == 'instance':
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
  elif layer_type == 'none':
    norm_layer = None
  else:
    raise NotImplementedError('normalization layer [%s] is not found' % layer_type)
  return norm_layer

def get_non_linearity(layer_type='relu'):
  if layer_type == 'relu':
    nl_layer = functools.partial(nn.ReLU, inplace=True)
  elif layer_type == 'lrelu':
    nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=False)
  elif layer_type == 'elu':
    nl_layer = functools.partial(nn.ELU, inplace=True)
  else:
    raise NotImplementedError('nonlinearity activitation [%s] is not found' % layer_type)
  return nl_layer
def conv3x3(in_planes, out_planes):
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)

def gaussian_weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1 and classname.find('Conv') == 0:
    m.weight.data.normal_(0.0, 0.02)

# ----------- basic blocks ----------------
class AdainNorm2d(nn.Module):
  def __init__(self, n_out, eps=1e-5, momentum=0.1):
    super(AdainNorm2d, self).__init__()
    self.n_out = n_out
    self.eps = eps
    self.momentum = momentum
    self.weight = None
    self.bias = None
    self.register_buffer('running_mean', torch.zeros(n_out))
    self.register_buffer('running_var', torch.ones(n_out))
    return
  def forward(self, x):
    assert self.weight is not None and self.bias is not None, "AdaNorm2d not initialized"
    b, c = x.size(0), x.size(1)
    running_mean = self.running_mean.repeat(b)
    running_var = self.running_var.repeat(b)
    x = x.contiguous().view(1, b*c, *x.size()[2:])
    out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps)
    return out.view(b, c, *x.size()[2:])
  def __repr__(self):
    return self.__class__.__name__ + '(' + str(self.n_out) + ')'

class LayerNorm(nn.Module):
  def __init__(self, n_out, eps=1e-5, affine=True):
    super(LayerNorm, self).__init__()
    self.n_out = n_out
    self.affine = affine
    self.eps = eps
    if self.affine:
      self.gamma = nn.Parameter(torch.Tensor(n_out).uniform_())
      self.beta = nn.Parameter(torch.zeros(n_out))
    return
  def forward(self, x):
    shape = [-1] + [1]*(x.dim() - 1)
    mean = x.view(x.size(0), -1).mean(1).view(*shape)
    std = x.view(x.size(0), -1).std(1).view(*shape)
    x = (x - mean) / (std + self.eps)
    if self.affine:
      shape = [1, -1] + [1]*(x.dim() - 2)
      x = x*self.gamma.view(*shape) + self.beta.view(*shape)
    return x

class BasicBlock(nn.Module):
  def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
    super(BasicBlock, self).__init__()
    layers = []
    if norm_layer is not None:
      layers += [norm_layer(inplanes)]
    layers += [nl_layer()]
    layers += [conv3x3(inplanes, inplanes)]
    if norm_layer is not None:
      layers += [norm_layer(inplanes)]
    layers += [nl_layer()]
    layers += [convMeanpool(inplanes, outplanes)]
    self.conv = nn.Sequential(*layers)
    self.shortcut = meanpoolConv(inplanes, outplanes)
  def forward(self, x):
    out = self.conv(x) + self.shortcut(x)
    return out

class Spectral_LeakyReLUConv2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
    super(Spectral_LeakyReLUConv2d, self).__init__()
    model = []
    model += [spectral_norm(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True))]
    model += [nn.LeakyReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
  def forward(self, x):
    return self.model(x)

class LeakyReLUConv2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
    super(LeakyReLUConv2d, self).__init__()
    model = []
    model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
    model += [nn.LeakyReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
  def forward(self, x):
    return self.model(x)

class ReLUINSConv2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
    super(ReLUINSConv2d, self).__init__()
    model = []
    model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
    model += [nn.InstanceNorm2d(n_out, affine=False)]
    model += [nn.ReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
  def forward(self, x):
    return self.model(x)

class INSResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1):
    return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)
  def __init__(self, inplanes, planes, stride=1, dropout=0.0):
    super(INSResBlock, self).__init__()
    model = []
    model += [self.conv3x3(inplanes, planes, stride)]
    model += [nn.InstanceNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    model += [self.conv3x3(planes, planes)]
    model += [nn.InstanceNorm2d(planes)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
  def forward(self, x):
    residual = x
    out = self.model(x)
    out += residual
    return out

class MisINSResBlock(nn.Module):
  def conv3x3(self, dim_in, dim_out, stride=1):
    return nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride, padding=1)
  def conv1x1(self, dim_in, dim_out):
    return nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)
  def __init__(self, dim, dim_extra, stride=1, dropout=0.0):
    super(MisINSResBlock, self).__init__()
    self.conv1 = nn.Sequential(
        self.conv3x3(dim, dim, stride),
        nn.InstanceNorm2d(dim))
    self.conv2 = nn.Sequential(
        self.conv3x3(dim, dim, stride),
        nn.InstanceNorm2d(dim))
    self.blk1 = nn.Sequential(
        self.conv1x1(dim + dim_extra, dim + dim_extra),
        nn.ReLU(inplace=False),
        self.conv1x1(dim + dim_extra, dim),
        nn.ReLU(inplace=False))
    self.blk2 = nn.Sequential(
        self.conv1x1(dim + dim_extra, dim + dim_extra),
        nn.ReLU(inplace=False),
        self.conv1x1(dim + dim_extra, dim),
        nn.ReLU(inplace=False))
    model = []
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
    self.conv1.apply(gaussian_weights_init)
    self.conv2.apply(gaussian_weights_init)
    self.blk1.apply(gaussian_weights_init)
    self.blk2.apply(gaussian_weights_init)
  def forward(self, x, z):
    residual = x
    z_expand = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
    o1 = self.conv1(x)
    o2 = self.blk1(torch.cat([o1, z_expand], dim=1))
    o3 = self.conv2(o2)
    out = self.blk2(torch.cat([o3, z_expand], dim=1))
    out += residual
    return out

class ADAINResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1):
    return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)
  def __init__(self, inplanes, planes, stride=1, dropout=0.0):
    super(ADAINResBlock, self).__init__()
    model = []
    model += [self.conv3x3(inplanes, planes, stride)]
    model += [AdainNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    model += [self.conv3x3(planes, planes)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
  def forward(self, x):
    residual = x
    out = self.model(x)
    out += residual
    return out

class GaussianNoiseLayer(nn.Module):
  def __init__(self,):
    super(GaussianNoiseLayer, self).__init__()
  def forward(self, x):
    if self.training == False:
      return x
    noise = Variable(torch.randn(x.size()).cuda(x.data.get_device()))
    return x + noise

class ReLUINSConvTranspose2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding, output_padding):
    super(ReLUINSConvTranspose2d, self).__init__()
    model = []
    model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=True)]
    model += [LayerNorm(n_out)]
    model += [nn.ReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
  def forward(self, x):
    return self.model(x)

# ----------- subnetwork for DRIT ----------------
class Dis_content(nn.Module):
  def __init__(self):
    super(Dis_content, self).__init__()
    model = []
    model += [LeakyReLUConv2d(256, 256, kernel_size=7, stride=2, padding=1)]
    model += [LeakyReLUConv2d(256, 256, kernel_size=7, stride=2, padding=1)]
    model += [LeakyReLUConv2d(256, 256, kernel_size=7, stride=2, padding=1)]
    model += [LeakyReLUConv2d(256, 256, kernel_size=4, stride=1, padding=0)]
    model += [nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)]
    self.model = nn.Sequential(*model)
  def forward(self, x):
    out = self.model(x)
    out = out.view(-1)
    outs = []
    outs.append(out)
    return outs

class MultiScaleDis(nn.Module):
  def __init__(self, input_dim, n_scale=3, n_layer=4):
    super(MultiScaleDis, self).__init__()
    ch = 64
    self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
    self.Diss = nn.ModuleList()
    for _ in range(n_scale):
      self.Diss.append(self._make_net(ch, input_dim, n_layer))

  def _make_net(self, ch, input_dim, n_layer):
    model = []
    model += [LeakyReLUConv2d(input_dim, ch, 4, 2, 1)]
    tch = ch
    for _ in range(1, n_layer):
      model += [LeakyReLUConv2d(tch, tch * 2, 4, 2, 1)]
      tch *= 2
    model += [nn.Conv2d(tch, 1, 1, 1, 0)]
    return nn.Sequential(*model)

  def forward(self, x):
    outs = []
    for Dis in self.Diss:
      outs.append(Dis(x))
      x = self.downsample(x)
    return outs

class Dis(nn.Module):
  def __init__(self, input_dim):
    super(Dis, self).__init__()
    ch = 64
    n_layer = 6
    self.model_A = self._make_net(ch, input_dim, n_layer)

  def _make_net(self, ch, input_dim, n_layer):
    model = []
    model += [Spectral_LeakyReLUConv2d(input_dim, ch, kernel_size=3, stride=2, padding=1)] #16
    tch = ch
    for i in range(1, n_layer):
      model += [Spectral_LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)] # 8
      tch *= 2
    model += [spectral_norm(nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0))]  # 1
    return nn.Sequential(*model)

  def cuda(self,gpu):
    self.model_A.cuda(gpu)

  def forward(self, x_A):
    out_A = self.model_A(x_A)
    out_A = out_A.view(-1)
    outs_A = []
    outs_A.append(out_A)
    return outs_A

class E_content(nn.Module):
  def __init__(self, input_dim_a, input_dim_b):
    super(E_content, self).__init__()
    encA_c = []
    tch = 64
    encA_c += [LeakyReLUConv2d(input_dim_a, tch, kernel_size=7, stride=1, padding=3)]
    for i in range(1,3):
      encA_c += [ReLUINSConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
      tch *= 2
    for i in range(0, 3):
      encA_c += [INSResBlock(tch, tch)]

    encB_c = []
    tch = 64
    encB_c += [LeakyReLUConv2d(input_dim_b, tch, kernel_size=7, stride=1, padding=3)]
    for i in range(1,3):
      encB_c += [ReLUINSConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
      tch *= 2
    for i in range(0, 3):
      encB_c += [INSResBlock(tch, tch)]

    enc_share = []
    for i in range(0, 1):
      enc_share += [INSResBlock(tch, tch)]
      enc_share += [GaussianNoiseLayer()]
      self.conv_share = nn.Sequential(*enc_share)

    self.convA = nn.Sequential(*encA_c)
    self.convB = nn.Sequential(*encB_c)

  def forward(self, xa, xb):
    outputA = self.convA(xa)
    outputB = self.convB(xb)
    outputA = self.conv_share(outputA)
    outputB = self.conv_share(outputB)
    return outputA, outputB

class E_attr(nn.Module):
  def __init__(self, input_dim_a, input_dim_b, output_nc=8):
    super(E_attr, self).__init__()
    dim = 64
    self.model_a = nn.Sequential(
        nn.Conv2d(input_dim_a, dim, 7, 1, 3),
        nn.ReLU(inplace=True),
        nn.Conv2d(dim, dim*2, 4, 2, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(dim*2, dim*4, 4, 2, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(dim*4, dim*4, 4, 2, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(dim*4, dim*4, 4, 2, 1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(dim*4, output_nc, 1, 1, 0))
    self.model_b = nn.Sequential(
        nn.Conv2d(input_dim_a, dim, 7, 1, 3),
        nn.ReLU(inplace=True),
        nn.Conv2d(dim, dim*2, 4, 2, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(dim*2, dim*4, 4, 2, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(dim*4, dim*4, 4, 2, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(dim*4, dim*4, 4, 2, 1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(dim*4, output_nc, 1, 1, 0))
    return
  def forward(self, xa, xb):
    xa = self.model_a(xa)
    xb = self.model_b(xb)
    output_A = xa.view(xa.size(0), -1)
    output_B = xb.view(xb.size(0), -1)
    return output_A, output_B

class E_attr_concat(nn.Module):
  def __init__(self, input_dim_a, input_dim_b, output_nc=8, norm_layer=None, nl_layer=None):
    super(E_attr_concat, self).__init__()

    ndf = 64
    n_blocks=4
    max_ndf = 4

    conv_layers_A = [nn.Conv2d(input_dim_a, ndf, kernel_size=4, stride=2, padding=1, bias=True)]
    for n in range(1, n_blocks):
      input_ndf = ndf * min(max_ndf, n)  # 2**(n-1)
      output_ndf = ndf * min(max_ndf, n+1)  # 2**n
      conv_layers_A += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer)]
    conv_layers_A += [nl_layer(), nn.AdaptiveAvgPool2d(1)] # AvgPool2d(13)
    self.fc_A = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
    self.fcVar_A = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
    self.conv_A = nn.Sequential(*conv_layers_A)

    conv_layers_B = [nn.Conv2d(input_dim_b, ndf, kernel_size=4, stride=2, padding=1, bias=True)]
    for n in range(1, n_blocks):
      input_ndf = ndf * min(max_ndf, n)  # 2**(n-1)
      output_ndf = ndf * min(max_ndf, n+1)  # 2**n
      conv_layers_B += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer)]
    conv_layers_B += [nl_layer(), nn.AdaptiveAvgPool2d(1)] # AvgPool2d(13)
    self.fc_B = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
    self.fcVar_B = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
    self.conv_B = nn.Sequential(*conv_layers_B)

  def forward(self, xa, xb):
    x_conv_A = self.conv_A(xa)
    conv_flat_A = x_conv_A.view(xa.size(0), -1)
    output_A = self.fc_A(conv_flat_A)
    outputVar_A = self.fcVar_A(conv_flat_A)
    x_conv_B = self.conv_B(xb)
    conv_flat_B = x_conv_B.view(xb.size(0), -1)
    output_B = self.fc_B(conv_flat_B)
    outputVar_B = self.fcVar_B(conv_flat_B)
    return output_A, outputVar_A, output_B, outputVar_B

class G(nn.Module):
  def __init__(self, output_dim_a, output_dim_b, nz):
    super(G, self).__init__()
    self.nz = nz
    ini_tch = 256
    tch_add = ini_tch
    self.tch_add = tch_add
    self.decA1 = MisINSResBlock(ini_tch, tch_add)
    self.decA2 = MisINSResBlock(ini_tch, tch_add)
    self.decA3 = MisINSResBlock(ini_tch, tch_add)
    self.decA4 = MisINSResBlock(ini_tch, tch_add)
    tch = ini_tch
    decA5 = []
    decA5 += [ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
    tch = tch//2
    decA5 += [ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
    tch = tch//2
    decA5 += [nn.ConvTranspose2d(tch, output_dim_a, kernel_size=1, stride=1, padding=0)]
    decA5 += [nn.Tanh()]
    self.decA5 = nn.Sequential(*decA5)

    tch = ini_tch
    self.decB1 = MisINSResBlock(tch, tch_add)
    self.decB2 = MisINSResBlock(tch, tch_add)
    self.decB3 = MisINSResBlock(tch, tch_add)
    self.decB4 = MisINSResBlock(tch, tch_add)
    decB5 = []
    decB5 += [ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
    tch = tch//2
    decB5 += [ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
    tch = tch//2
    decB5 += [nn.ConvTranspose2d(tch, output_dim_b, kernel_size=1, stride=1, padding=0)]
    decB5 += [nn.Tanh()]
    self.decB5 = nn.Sequential(*decB5)

    self.mlpA = nn.Sequential(
        nn.Linear(8, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, tch_add*4))
    self.mlpB = nn.Sequential(
        nn.Linear(8, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, tch_add*4))
    return
  def forward_a(self, x, z):
    z = self.mlpA(z)
    z1, z2, z3, z4 = torch.split(z, self.tch_add, dim=1)
    z1 = z1.contiguous()
    z2 = z2.contiguous()
    z3 = z3.contiguous()
    z4 = z4.contiguous()
    out1 = self.decA1(x, z1)
    out2 = self.decA2(out1, z2)
    out3 = self.decA3(out2, z3)
    out4 = self.decA4(out3, z4)
    out = self.decA5(out4)
    return out
  def forward_b(self, x, z):
    z = self.mlpB(z)
    z1, z2, z3, z4 = torch.split(z, self.tch_add, dim=1)
    z1 = z1.contiguous()
    z2 = z2.contiguous()
    z3 = z3.contiguous()
    z4 = z4.contiguous()
    out1 = self.decB1(x, z1)
    out2 = self.decB2(out1, z2)
    out3 = self.decB3(out2, z3)
    out4 = self.decB4(out3, z4)
    out = self.decB5(out4)
    return out

class G_concat(nn.Module):
  def __init__(self, output_dim_a, output_dim_b, nz):
    super(G_concat, self).__init__()
    self.nz = nz
    tch = 256
    dec_share = []
    dec_share += [INSResBlock(tch, tch)]
    self.dec_share = nn.Sequential(*dec_share)
    tch = 256+self.nz
    decA1 = []
    for i in range(0, 3):
      decA1 += [INSResBlock(tch, tch)]
    tch = tch + self.nz
    decA2 = ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)
    tch = tch//2
    tch = tch + self.nz
    decA3 = ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)
    tch = tch//2
    tch = tch + self.nz
    decA4 = [nn.ConvTranspose2d(tch, output_dim_a, kernel_size=1, stride=1, padding=0)]+[nn.Tanh()]
    self.decA1 = nn.Sequential(*decA1)
    self.decA2 = nn.Sequential(*[decA2])
    self.decA3 = nn.Sequential(*[decA3])
    self.decA4 = nn.Sequential(*decA4)

    tch = 256+self.nz
    decB1 = []
    for i in range(0, 3):
      decB1 += [INSResBlock(tch, tch)]
    tch = tch + self.nz
    decB2 = ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)
    tch = tch//2
    tch = tch + self.nz
    decB3 = ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)
    tch = tch//2
    tch = tch + self.nz
    decB4 = [nn.ConvTranspose2d(tch, output_dim_b, kernel_size=1, stride=1, padding=0)]+[nn.Tanh()]
    self.decB1 = nn.Sequential(*decB1)
    self.decB2 = nn.Sequential(*[decB2])
    self.decB3 = nn.Sequential(*[decB3])
    self.decB4 = nn.Sequential(*decB4)

  def forward_a(self, x, z):
    out0 = self.dec_share(x)
    z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
    x_and_z = torch.cat([out0, z_img], 1)
    out1 = self.decA1(x_and_z)
    z_img2 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out1.size(2), out1.size(3))
    x_and_z2 = torch.cat([out1, z_img2], 1)
    out2 = self.decA2(x_and_z2)
    z_img3 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out2.size(2), out2.size(3))
    x_and_z3 = torch.cat([out2, z_img3], 1)
    out3 = self.decA3(x_and_z3)
    z_img4 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out3.size(2), out3.size(3))
    x_and_z4 = torch.cat([out3, z_img4], 1)
    out4 = self.decA4(x_and_z4)
    return out4
  def forward_b(self, x, z):
    out0 = self.dec_share(x)
    z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
    x_and_z = torch.cat([out0,  z_img], 1)
    out1 = self.decB1(x_and_z)
    z_img2 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out1.size(2), out1.size(3))
    x_and_z2 = torch.cat([out1, z_img2], 1)
    out2 = self.decB2(x_and_z2)
    z_img3 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out2.size(2), out2.size(3))
    x_and_z3 = torch.cat([out2, z_img3], 1)
    out3 = self.decB3(x_and_z3)
    z_img4 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out3.size(2), out3.size(3))
    x_and_z4 = torch.cat([out3, z_img4], 1)
    out4 = self.decB4(x_and_z4)
    return out4
