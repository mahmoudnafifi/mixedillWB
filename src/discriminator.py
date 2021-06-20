from torch import nn
from math import log2

class DiscriminatorBlock(nn.Module):
  def __init__(self, input_channels, filters, downsample=True):
    super().__init__()
    self.conv_res = nn.Conv2d(input_channels, filters, 1)

    self.net = nn.Sequential(
      nn.Conv2d(input_channels, filters, 3, padding=1),
      nn.LeakyReLU(inplace=False),
      nn.Conv2d(filters, filters, 3, padding=1),
      nn.LeakyReLU(inplace=False),
    )

    self.downsample = nn.Conv2d(filters, filters, 3, padding=1,
                                stride=2) if downsample else None

  def forward(self, x):
    res = self.conv_res(x)
    x = self.net(x)
    x = x + res
    if self.downsample is not None:
      x = self.downsample(x)
    return x



class Discriminator(nn.Module):
  def __init__(self, image_size, network_capacity=12):
    super().__init__()
    num_layers = int(log2(image_size) - 1)
    num_init_filters = 3
    blocks = []
    filters = [num_init_filters] + [(network_capacity) * (2 ** i) for i in
                                    range(num_layers + 1)]
    chan_in_out = list(zip(filters[0:-1], filters[1:]))

    for ind, (in_chan, out_chan) in enumerate(chan_in_out):
      is_not_last = ind != (len(chan_in_out) - 1)

      block = DiscriminatorBlock(in_chan, out_chan, downsample=is_not_last)
      blocks.append(block)

    self.blocks = nn.ModuleList(blocks)

    latent_dim = 2 * 2 * filters[-1]

    self.flatten = Flatten()
    self.to_logit = nn.Linear(latent_dim, 1)


  def forward(self, x):
    b, *_ = x.shape
    for block in self.blocks:
      x = block(x)
    x = self.flatten(x)
    x = self.to_logit(x)
    return x.squeeze()


class Flatten(nn.Module):
  def forward(self, x):
    return x.reshape(x.shape[0], -1)