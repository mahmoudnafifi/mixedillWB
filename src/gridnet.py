"""
 A simple Pytorch implementation of GridNet, presented in Ref. 1. This
   implementation includes the modified version proposed in Ref. 2 (recommended
   for image-to-image translation).
 References:
   Ref. 1: Residual Conv-Deconv Grid Network for Semantic Segmentation,
     In BMVC, 2017.
   Ref. 2: Context-aware Synthesis for Video Frame Interpolation, In CVPR 2018.
"""

__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]


import torch.nn as nn

class network(nn.Module):
  def __init__(self, inchnls=3, outchnls=3, initialchnls=16, rows=3,
               columns=6, norm=False, device='cuda'):
    """ GridNet constructor.

    Args:
      inchnls: input channels; default is 3.
      outchnls: output channels; default is 3.
      initialchnls: initial number of feature channels; default is 16.
      rows: number of rows; default is 3.
      columns: number of columns; default is 6 (should be an even number).
      norm: apply batch norm as used in Ref. 1; default is False (i.e., Ref. 2)
    """

    super(network, self).__init__()
    assert columns % 2 == 0, 'use even number of columns'
    assert columns > 1, 'use number of columns > 1'
    assert rows > 1, 'use number of rows > 1'

    self.device = device

    self.encoder = nn.ModuleList([])
    self.decoder = nn.ModuleList([])
    self.rows = rows
    self.columns = columns

    # encoder
    for r in range(rows):
      res_blocks = nn.ModuleList([])
      down_blocks = nn.ModuleList([])
      for c in range(int(columns / 2)):
        if r == 0:
          if c == 0:
            res_blocks.append(ForwardBlock(in_dim=inchnls,
                                            out_dim=initialchnls,
                                            norm=norm).to(device=self.device))
          else:
            res_blocks.append(ResidualBlock(in_dim=initialchnls, norm=norm).to(
              device=self.device))
          down_blocks.append(SubsamplingBlock(
            in_dim=initialchnls, norm=norm).to(device=self.device))
        else:
          if c > 0:
            res_blocks.append(ResidualBlock(
              in_dim=initialchnls * (2 ** r), norm=norm).to(
              device=self.device))
          else:
            res_blocks.append(nn.ModuleList([]))
          if r < (rows - 1):
            down_blocks.append(SubsamplingBlock(
              in_dim=initialchnls * (2 ** r), norm=norm).to(
              device=self.device))
          else:
            down_blocks.append(nn.ModuleList([]))

      self.encoder.append(res_blocks)
      self.encoder.append(down_blocks)


    # decoder
    for r in range((rows - 1), -1, -1):
      res_blocks = nn.ModuleList([])
      up_blocks = nn.ModuleList([])
      for c in range(int(columns / 2), columns):
        if r == 0:
          res_blocks.append(ResidualBlock(in_dim=initialchnls,
                                          norm=norm).to(device=self.device))
          up_blocks.append(nn.ModuleList([]))
        elif r > 0:
          res_blocks.append(ResidualBlock(
              in_dim=initialchnls * (2 ** r), norm=norm).to(
            device=self.device))
          up_blocks.append(UpsamplingBlock(
            in_dim=initialchnls * (2 ** r), norm=norm).to(
            device=self.device))

      self.decoder.append(res_blocks)
      self.decoder.append(up_blocks)

    self.output = ForwardBlock(in_dim=initialchnls, out_dim=outchnls,
                                norm=norm).to(device=self.device)


  def forward(self, x):
    """ Forward function

    Args:
      x: input image

    Returns:
      output: output image
    """
    latent_downscaled = []
    latent_upscaled = []
    latent_forward = []

    for i in range(0, len(self.encoder), 2):
      res_blcks = self.encoder[i]
      branch_blcks = self.encoder[i + 1]
      if not branch_blcks[0]:
        not_last = False
      else:
        not_last = True
      for j, (res_blck, branch_blck) in enumerate(zip(res_blcks,
                                                      branch_blcks)):
        if i == 0 and j == 0:
          x_latent = res_blck(x)
        elif i == 0:
          x_latent = res_blck(x_latent)
        elif j == 0:
          x_latent = latent_downscaled[j]
        else:
          x_latent = res_blck(x_latent)
          x_latent = x_latent + latent_downscaled[j]
        if i == 0:
          latent_downscaled.append(branch_blck(x_latent))
        elif not_last:
          latent_downscaled[j] = branch_blck(x_latent)
      latent_forward.append(x_latent)

    latent_forward.reverse()

    for k, i in enumerate(range(0, len(self.decoder), 2)):
      res_blcks = self.decoder[i]
      branch_blcks = self.decoder[i + 1]
      if not branch_blcks[0]:
        not_last = False
      else:
        not_last = True
      for j, (res_blck, branch_blck) in enumerate(zip(res_blcks,
                                                      branch_blcks)):
        if j == 0:
          latent_x = latent_forward[k]
        x_latent = res_blck(latent_x)
        if i > 0:
          x_latent = x_latent + latent_upscaled[j]
        if i == 0:
          latent_upscaled.append(branch_blck(x_latent))
        elif not_last:
          latent_upscaled[j] = branch_blck(x_latent)

    output = self.output(x_latent)
    return output


class SubsamplingBlock(nn.Module):
  """ SubsamplingBlock"""

  def __init__(self, in_dim, norm=False):
    super(SubsamplingBlock, self).__init__()
    self.output = None
    if norm:
      self.block = nn.Sequential(
        nn.BatchNorm2d(in_dim),
        nn.PReLU(init=0.25),
        nn.Conv2d(in_dim, int(in_dim * 2), kernel_size=3, padding=1, stride=2),
        nn.BatchNorm2d(int(in_dim * 2)),
        nn.ReLU(inplace=True),
        nn.Conv2d(int(in_dim * 2), int(in_dim * 2), kernel_size=3, padding=1))
    else:
      self.block = nn.Sequential(
        nn.PReLU(init=0.25),
        nn.Conv2d(in_dim, int(in_dim * 2), kernel_size=3, padding=1, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(int(in_dim * 2), int(in_dim * 2), kernel_size=3, padding=1))

  def forward(self, x):
    return self.block(x)


class UpsamplingBlock(nn.Module):
  """ UpsamplingBlock"""

  def __init__(self, in_dim, norm=False):
    super(UpsamplingBlock, self).__init__()
    self.output = None
    if norm:
      self.block = nn.Sequential(
        nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
        nn.BatchNorm2d(in_dim),
        nn.PReLU(init=0.25),
        nn.Conv2d(in_dim, int(in_dim / 2), kernel_size=3, padding=1),
        nn.BatchNorm2d(int(in_dim / 2)),
        nn.ReLU(inplace=True),
        nn.Conv2d(int(in_dim / 2), int(in_dim / 2), kernel_size=3, padding=1))
    else:
      self.block = nn.Sequential(
        nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
        nn.PReLU(init=0.25),
        nn.Conv2d(in_dim, int(in_dim / 2), kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(int(in_dim / 2), int(in_dim / 2), kernel_size=3, padding=1))

  def forward(self, x):
    return self.block(x)


class ResidualBlock(nn.Module):
  """ ResidualBlock"""

  def __init__(self, in_dim, out_dim=None, norm=False):
    super(ResidualBlock, self).__init__()
    self.output = None
    intermediate_dim = int(in_dim * 2)
    if out_dim is None:
      out_dim = in_dim
    if norm:
      self.block = nn.Sequential(
        nn.BatchNorm2d(in_dim),
        nn.PReLU(init=0.25),
        nn.Conv2d(in_dim, intermediate_dim, kernel_size=3, padding=1),
        nn.BatchNorm2d(intermediate_dim),
        nn.PReLU(init=0.25),
        nn.Conv2d(intermediate_dim, out_dim, kernel_size=3, padding=1))
    else:
      self.block = nn.Sequential(
        nn.PReLU(init=0.25),
        nn.Conv2d(in_dim, intermediate_dim, kernel_size=3, padding=1),
        nn.PReLU(init=0.25),
        nn.Conv2d(intermediate_dim, out_dim, kernel_size=3, padding=1))

  def forward(self, x):
    return x + self.block(x)



class ForwardBlock(nn.Module):
  """ ForwardBlock"""

  def __init__(self, in_dim, out_dim=None, norm=False):
    super(ForwardBlock, self).__init__()
    self.output = None
    intermediate_dim = int(in_dim * 2)
    if out_dim is None:
      out_dim = in_dim
    if norm:
      self.block = nn.Sequential(
        nn.BatchNorm2d(in_dim),
        nn.PReLU(init=0.25),
        nn.Conv2d(in_dim, intermediate_dim, kernel_size=3, padding=1),
        nn.BatchNorm2d(intermediate_dim),
        nn.PReLU(init=0.25),
        nn.Conv2d(intermediate_dim, out_dim, kernel_size=3, padding=1))
    else:
      self.block = nn.Sequential(
        nn.PReLU(init=0.25),
        nn.Conv2d(in_dim, intermediate_dim, kernel_size=3, padding=1),
        nn.PReLU(init=0.25),
        nn.Conv2d(intermediate_dim, out_dim, kernel_size=3, padding=1))

  def forward(self, x):
    return self.block(x)




