from os.path import join
from os import listdir
from os import path
from torch.utils.data import Dataset
import logging
from src import ops
import torch
import numpy as np
from WB_aug.WBAugmenter import WBEmulator as wbAug


class Data(Dataset):
  def __init__(self, imgfiles, patch_size=128, patch_number=32, aug=True,
               shuffle_order=False, mode='training'):
    """ Data constructor

    Args:
      imgfiles: a list of full filenames to be used by the dataloader.
      patch_size: training patch dimension (int).

    Returns:
      Dataset loader object with the selected settings.
    """

    self.imgfiles = imgfiles
    self.patch_size = patch_size
    self.patch_number = patch_number
    self.aug = aug
    self.shuffle_order = shuffle_order
    assert (mode == 'training' or
            mode == 'testing'), 'mode should be training or testing'
    self.mode = mode

    if shuffle_order is True and self.mode == 'testing':
      logging.warning('Shuffling is not allowed in testing mode')
      self.shuffle_order = False


    if self.mode == 'testing':
      self.wbColorAug = wbAug.WBEmulator()

    logging.info(f'Creating dataset with {len(self.imgfiles)} examples')

  def __len__(self):
    """ Gets length of image files in the dataloader. """

    return len(self.imgfiles)

  def __getitem__(self, i):
    """ Gets next data in the dataloader.

    Args:
      i: index of file in the dataloader.

    Returns:
      A dictionary of the following keys:
      - image:
    """

    D_img_file = self.imgfiles[i]
    d_img = ops.imread(D_img_file)

    if self.mode == 'testing':
      full_size_img = d_img.copy()

    d_img = ops.imresize.imresize(d_img, output_shape=(320, 320))

    base_name = ops.get_basename(D_img_file)

    if self.mode == 'training':
      gt_img_file = path.basename(base_name) + 'G_AS.png'

      gt_img_file = path.join(path.split(path.dirname(D_img_file))[0],
                              'ground truth images', gt_img_file)

      gt_img = ops.imread(gt_img_file)

      gt_img = ops.imresize.imresize(gt_img, output_shape=(320, 320))

    if self.mode == 'training':
      s_img_file = base_name + 'S_CS.png'
      s_img = ops.imread(s_img_file)

      s_img = ops.imresize.imresize(s_img, output_shape=(320, 320))

      t_img_file = base_name + 'T_CS.png'
      t_img = ops.imread(t_img_file)

      t_img = ops.imresize.imresize(t_img, output_shape=(320, 320))

      if self.aug:
        d_img, s_img, t_img, gt_img = ops.aug(d_img, s_img, t_img, gt_img)

      d_img, s_img, t_img, gt_img = ops.extract_patch(
        d_img, s_img, t_img, gt_img, patch_size=self.patch_size,
        patch_number=self.patch_number)
      d_img = ops.to_tensor(d_img, dims=4)
      s_img = ops.to_tensor(s_img, dims=4)
      t_img = ops.to_tensor(t_img, dims=4)
      gt_img = ops.to_tensor(gt_img, dims=4)

      if self.shuffle_order:
        order_ind = np.random.randint(6)
        if order_ind == 0:
          img = torch.cat((d_img, s_img, t_img), dim=1)
          order = [0, 1, 2]
        elif order_ind == 1:
          img = torch.cat((d_img, t_img, s_img), dim=1)
          order = [0, 2, 1]
        elif order_ind == 2:
          img = torch.cat((s_img, d_img, t_img), dim=1)
          order = [1, 0, 2]
        elif order_ind == 3:
          img = torch.cat((s_img, t_img, d_img), dim=1)
          order = [1, 2, 0]
        elif order_ind == 4:
          img = torch.cat((t_img, d_img, s_img), dim=1)
          order = [2, 0, 1]
        elif order_ind == 5:
          img = torch.cat((t_img, s_img, d_img), dim=1)
          order = [2, 1, 0]
      else:
        img = torch.cat((d_img, s_img, t_img), dim=1)
        order = [0, 1, 2]

      return {'image': img, 'gt': gt_img, 'filename': base_name, 'order': order}

    else:  # testing mode
      s_img_file = base_name + 'S_CS.png'
      t_img_file = base_name + 'T_CS.png'
      if path.exists(s_img_file) and path.exists(t_img_file):
        s_img = ops.imread(s_img_file)
        s_img = ops.imresize.imresize(s_img, output_shape=(320, 320))
        t_img = ops.imread(t_img_file)
        t_img = ops.imresize.imresize(t_img, output_shape=(320, 320))
        s_mapping = ops.get_mapping_func(d_img, s_img)
        t_mapping = ops.get_mapping_func(d_img, t_img)
        full_size_s = ops.apply_mapping_func(full_size_img, s_mapping)
        full_size_s = ops.outOfGamutClipping(full_size_s)
        full_size_t = ops.apply_mapping_func(full_size_img, t_mapping)
        full_size_t = ops.outOfGamutClipping(full_size_t)
      else:
        base_name = D_img_file
        base_name = path.splitext(base_name)[0]
        outImgs, _ = self.wbColorAug.generateWbsRGB(d_img)
        s_img = outImgs[:, :, :, 3]
        s_img = ops.outOfGamutClipping(s_img)
        t_img = outImgs[:, :, :, 5]
        t_img = ops.outOfGamutClipping(t_img)

        s_mapping = ops.get_mapping_func(d_img, s_img)
        t_mapping = ops.get_mapping_func(d_img, t_img)
        full_size_s = ops.apply_mapping_func(full_size_img, s_mapping)
        full_size_s = ops.outOfGamutClipping(full_size_s)
        full_size_t = ops.apply_mapping_func(full_size_img, t_mapping)
        full_size_t = ops.outOfGamutClipping(full_size_t)

      d_img = ops.to_tensor(d_img, dims=3)
      s_img = ops.to_tensor(s_img, dims=3)
      t_img = ops.to_tensor(t_img, dims=3)
      img = torch.cat((d_img, s_img, t_img), dim=0)

      full_size_img = ops.to_tensor(full_size_img, dims=3)
      full_size_s = ops.to_tensor(full_size_s, dims=3)
      full_size_t = ops.to_tensor(full_size_t, dims=3)

      return {'image': img, 'fs_d_img': full_size_img, 'fs_s_img': full_size_s,
              'fs_t_img': full_size_t, 'filename': base_name}


  @staticmethod
  def load_files(img_dir, mode='training'):
    """ Loads filenames in a given image directory.

    Args:
      img_dir: image directory.

    Returns:
      imgfiles: a list of full filenames.
    """
    if mode == 'training':
      ext = ['_D_CS.png', '_D_CS.PNG']
    else:
      ext = ['.png', '.PNG', '.jpg', '.JPG']
    logging.info(f'Loading images information from {img_dir}...')
    if mode == 'training':
      imgfiles = [join(img_dir, file) for file in listdir(img_dir)
                  if file.endswith(ext[0]) or file.endswith(ext[1])]
    else:
      imgfiles = [join(img_dir, file) for file in listdir(img_dir)
                  if file.endswith(ext[0]) or file.endswith(ext[1]) or
                  file.endswith(ext[2]) or file.endswith(ext[3])]
    return imgfiles

  @staticmethod
  def assert_files(files):
    for file in files:
      base_name = ops.get_basename(file)
      gt_img_file = path.basename(base_name) + 'G_AS.png'
      gt_img_file = path.join(path.split(path.dirname(file))[0],
                              'ground truth images', gt_img_file)
      s_img_file = base_name + 'S_CS.png'
      t_img_file = base_name + 'T_CS.png'
      assert (path.exists(file) and path.exists(gt_img_file) and path.exists(
        s_img_file) and path.exists(t_img_file))

    return True
