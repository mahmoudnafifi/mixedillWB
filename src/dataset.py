from os.path import join
from os import listdir
from os import path
from torch.utils.data import Dataset
import logging
from src import ops
import torch
import numpy as np
from DeepWB.arch import deep_wb_single_task as dwb
from DeepWB.utilities.deepWB import deep_wb
from DeepWB.utilities.utils import colorTempInterpolate_w_target


class Data(Dataset):
  def __init__(self, imgfiles, patch_size=128, patch_number=32, aug=True,
               wb_settings=None, shuffle_order=False, mode='training',
               multiscale=False, keep_aspect_ratio=False, t_size=320):
    """ Data constructor
    """

    if wb_settings is None:
      self.wb_settings = ['D', 'T', 'F', 'C', 'S']
    else:
      self.wb_settings = wb_settings
    assert ('S' in self.wb_settings and 'T' in self.wb_settings and 'D' in
           self.wb_settings), 'Incorrect WB settings'

    for wb_setting in self.wb_settings:
      assert wb_setting in ['D', 'T', 'F', 'C', 'S']

    self.imgfiles = imgfiles
    self.patch_size = patch_size
    self.patch_number = patch_number
    self.keep_aspect_ratio = keep_aspect_ratio
    self.aug = aug
    self.multiscale = multiscale
    self.shuffle_order = shuffle_order
    assert (mode == 'training' or
            mode == 'testing'), 'mode should be training or testing'
    self.mode = mode

    if shuffle_order is True and self.mode == 'testing':
      logging.warning('Shuffling is not allowed in testing mode')
      self.shuffle_order = False

    self.t_size = t_size

    if self.mode == 'testing':
      self.deepWB_T = dwb.deepWBnet()
      self.deepWB_T.load_state_dict(torch.load('DeepWB/models/net_t.pth'))
      self.deepWB_S = dwb.deepWBnet()
      self.deepWB_S.load_state_dict(torch.load('DeepWB/models/net_s.pth'))
      self.deepWB_T.eval().to(device='cuda')
      self.deepWB_S.eval().to(device='cuda')


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
      t_size = self.t_size
      full_size_img = d_img.copy()

    base_name = ops.get_basename(D_img_file)

    if self.mode == 'training':
      if self.multiscale:
        t_size = self.t_size + 64 * 2 ** np.random.randint(5)
      else:
        t_size = self.t_size

      d_img = ops.imresize.imresize(d_img, output_shape=(t_size, t_size))

      gt_img_file = path.basename(base_name) + 'G_AS.png'

      gt_img_file = path.join(path.split(path.dirname(D_img_file))[0],
                              'ground truth images', gt_img_file)

      gt_img = ops.imread(gt_img_file)

      gt_img = ops.imresize.imresize(gt_img, output_shape=(t_size, t_size))

      s_img_file = base_name + 'S_CS.png'
      s_img = ops.imread(s_img_file)

      s_img = ops.imresize.imresize(s_img, output_shape=(t_size, t_size))

      t_img_file = base_name + 'T_CS.png'
      t_img = ops.imread(t_img_file)

      t_img = ops.imresize.imresize(t_img, output_shape=(t_size, t_size))

      if 'F' in self.wb_settings:
        f_img_file = base_name + 'F_CS.png'
        f_img = ops.imread(f_img_file)
        f_img = ops.imresize.imresize(f_img, output_shape=(t_size, t_size))
      else:
        f_img = None

      if 'C' in self.wb_settings:
        c_img_file = base_name + 'C_CS.png'
        c_img = ops.imread(c_img_file)
        c_img = ops.imresize.imresize(c_img, output_shape=(t_size, t_size))
      else:
        c_img = None


      if self.aug:
        if f_img is not None and c_img is not None:
          d_img, s_img, t_img, f_img, c_img, gt_img = ops.aug(
            d_img, s_img, t_img, f_img, c_img, gt_img)
        elif f_img is not None:
          d_img, s_img, t_img, f_img, gt_img = ops.aug(
            d_img, s_img, t_img, f_img, gt_img)
        elif c_img is not None:
          d_img, s_img, t_img, c_img, gt_img = ops.aug(
            d_img, s_img, t_img, c_img, gt_img)
        else:
          d_img, s_img, t_img, gt_img = ops.aug(d_img, s_img, t_img, gt_img)

      if f_img is not None and c_img is not None:
        d_img, s_img, t_img, f_img, c_img, gt_img = ops.extract_patch(
          d_img, s_img, t_img, f_img, c_img, gt_img, patch_size=self.patch_size,
          patch_number=self.patch_number)
      elif f_img is not None:
        d_img, s_img, t_img, f_img, gt_img = ops.extract_patch(
          d_img, s_img, t_img, f_img, gt_img, patch_size=self.patch_size,
          patch_number=self.patch_number)
      elif c_img is not None:
        d_img, s_img, t_img, c_img, gt_img = ops.extract_patch(
          d_img, s_img, t_img, c_img, gt_img, patch_size=self.patch_size,
          patch_number=self.patch_number)
      else:
        d_img, s_img, t_img, gt_img = ops.extract_patch(
          d_img, s_img, t_img, gt_img, patch_size=self.patch_size,
          patch_number=self.patch_number)

      d_img = ops.to_tensor(d_img, dims=3 + int(self.aug))
      s_img = ops.to_tensor(s_img, dims=3 + int(self.aug))
      t_img = ops.to_tensor(t_img, dims=3 + int(self.aug))
      gt_img = ops.to_tensor(gt_img, dims=3 + int(self.aug))
      if f_img is not None:
        f_img = ops.to_tensor(f_img, dims=3 + int(self.aug))
      if c_img is not None:
        c_img = ops.to_tensor(c_img, dims=3 + int(self.aug))

      if self.shuffle_order:
        imgs = [d_img, s_img, t_img]
        if f_img is not None:
          imgs.append(f_img)
        if c_img is not None:
          imgs.append(c_img)
        order = np.random.permutation(len(imgs))

        img = torch.cat((imgs[order[0]], imgs[order[1]], imgs[order[2]]), dim=1)
        for i in range(3, len(imgs), 1):
          img = torch.cat((img, imgs[order[i]]), dim=1)

      else:
        img = torch.cat((d_img, s_img, t_img), dim=1)
        if f_img is not None:
          img = torch.cat((img, f_img), dim=1)
        if c_img is not None:
          img = torch.cat((img, c_img), dim=1)

      return {'image': img, 'gt': gt_img, 'filename': base_name}

    else:  # testing mode
      s_img_file = base_name + 'S_CS.png'
      t_img_file = base_name + 'T_CS.png'
      paths = [s_img_file, t_img_file]
      if 'F' in self.wb_settings:
        f_img_file = base_name + 'F_CS.png'
        paths.append(f_img_file)
      if 'C' in self.wb_settings:
        c_img_file = base_name + 'C_CS.png'
        paths.append(c_img_file)

      checks = True
      for curr_path in paths:
        checks = checks & path.exists(curr_path)

      if checks:
        if self.keep_aspect_ratio:
          d_img = ops.aspect_ratio_imresize(d_img, max_output=t_size)
        else:
          d_img = ops.imresize.imresize(d_img, output_shape=(t_size, t_size))
        s_img = ops.imread(s_img_file)
        if self.keep_aspect_ratio:
          s_img = ops.aspect_ratio_imresize(s_img, max_output=t_size)
        else:
          s_img = ops.imresize.imresize(s_img, output_shape=(t_size, t_size))
        s_mapping = ops.get_mapping_func(d_img, s_img)
        full_size_s = ops.apply_mapping_func(full_size_img, s_mapping)
        full_size_s = ops.outOfGamutClipping(full_size_s)

        t_img = ops.imread(t_img_file)
        if self.keep_aspect_ratio:
          t_img = ops.aspect_ratio_imresize(t_img, max_output=t_size)
        else:
          t_img = ops.imresize.imresize(t_img, output_shape=(t_size, t_size))
        t_mapping = ops.get_mapping_func(d_img, t_img)
        full_size_t = ops.apply_mapping_func(full_size_img, t_mapping)
        full_size_t = ops.outOfGamutClipping(full_size_t)

        if 'F' in self.wb_settings:
          f_img = ops.imread(f_img_file)
          if self.keep_aspect_ratio:
            f_img = ops.aspect_ratio_imresize(f_img, max_output=t_size)
          else:
            f_img = ops.imresize.imresize(f_img, output_shape=(t_size, t_size))
          f_mapping = ops.get_mapping_func(d_img, f_img)
          full_size_f = ops.apply_mapping_func(full_size_img, f_mapping)
          full_size_f = ops.outOfGamutClipping(full_size_f)
        else:
          f_img = None

        if 'C' in self.wb_settings:
          c_img = ops.imread(c_img_file)
          if self.keep_aspect_ratio:
            c_img = ops.aspect_ratio_imresize(c_img, max_output=t_size)
          else:
            c_img = ops.imresize.imresize(c_img, output_shape=(t_size, t_size))
          c_mapping = ops.get_mapping_func(d_img, c_img)
          full_size_c = ops.apply_mapping_func(full_size_img, c_mapping)
          full_size_c = ops.outOfGamutClipping(full_size_c)
        else:
          c_img = None

      else:
        base_name = D_img_file
        base_name = path.splitext(base_name)[0]
        t_img, s_img = deep_wb(d_img, task='editing', net_s=self.deepWB_S,
                               net_t=self.deepWB_T, device='cuda')
        if self.keep_aspect_ratio:
          d_img = ops.aspect_ratio_imresize(d_img, max_output=t_size)
          t_img = ops.aspect_ratio_imresize(t_img, max_output=t_size)
          s_img = ops.aspect_ratio_imresize(s_img, max_output=t_size)
        else:
          d_img = ops.imresize.imresize(d_img, output_shape=(t_size, t_size))
          t_img = ops.imresize.imresize(t_img, output_shape=(t_size, t_size))
          s_img = ops.imresize.imresize(s_img, output_shape=(t_size, t_size))

        s_mapping = ops.get_mapping_func(d_img, s_img)
        t_mapping = ops.get_mapping_func(d_img, t_img)
        full_size_s = ops.apply_mapping_func(full_size_img, s_mapping)
        full_size_s = ops.outOfGamutClipping(full_size_s)
        full_size_t = ops.apply_mapping_func(full_size_img, t_mapping)
        full_size_t = ops.outOfGamutClipping(full_size_t)

        if 'F' in self.wb_settings:
          f_img = colorTempInterpolate_w_target(t_img, s_img, 3800)
          f_mapping = ops.get_mapping_func(d_img, f_img)
          full_size_f = ops.apply_mapping_func(full_size_img, f_mapping)
          full_size_f = ops.outOfGamutClipping(full_size_f)
        else:
          f_img = None

        if 'C' in self.wb_settings:
          c_img = colorTempInterpolate_w_target(t_img, s_img,  6500)
          c_mapping = ops.get_mapping_func(d_img, c_img)
          full_size_c = ops.apply_mapping_func(full_size_img, c_mapping)
          full_size_c = ops.outOfGamutClipping(full_size_c)
        else:
          c_img = None


      d_img = ops.to_tensor(d_img, dims=3)
      s_img = ops.to_tensor(s_img, dims=3)
      t_img = ops.to_tensor(t_img, dims=3)

      if f_img is not None:
        f_img = ops.to_tensor(f_img, dims=3)
      if c_img is not None:
        c_img = ops.to_tensor(c_img, dims=3)

      img = torch.cat((d_img, s_img, t_img), dim=0)
      if f_img is not None:
        img = torch.cat((img, f_img), dim=0)
      if c_img is not None:
        img = torch.cat((img, c_img), dim=0)

      full_size_img = ops.to_tensor(full_size_img, dims=3)
      full_size_s = ops.to_tensor(full_size_s, dims=3)
      full_size_t = ops.to_tensor(full_size_t, dims=3)

      if c_img is not None:
        full_size_c = ops.to_tensor(full_size_c, dims=3)

      if f_img is not None:
        full_size_f = ops.to_tensor(full_size_f, dims=3)

      if c_img is not None and f_img is not None:
        return {'image': img, 'fs_d_img': full_size_img, 'fs_s_img':
          full_size_s, 'fs_t_img': full_size_t, 'fs_f_img': full_size_f,
                'fs_c_img': full_size_c, 'filename': base_name}
      elif c_img is not None:
        return {'image': img, 'fs_d_img': full_size_img, 'fs_s_img':
          full_size_s, 'fs_t_img': full_size_t, 'fs_c_img': full_size_c,
                'filename': base_name}
      elif f_img is not None:
        return {'image': img, 'fs_d_img': full_size_img, 'fs_s_img':
          full_size_s, 'fs_t_img': full_size_t, 'fs_f_img': full_size_f,
                'filename': base_name}
      else:
        return {'image': img, 'fs_d_img': full_size_img, 'fs_s_img':
          full_size_s, 'fs_t_img': full_size_t, 'filename': base_name}


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
  def assert_files(files, wb_settings):
    for file in files:
      base_name = ops.get_basename(file)
      gt_img_file = path.basename(base_name) + 'G_AS.png'
      gt_img_file = path.join(path.split(path.dirname(file))[0],
                              'ground truth images', gt_img_file)
      s_img_file = base_name + 'S_CS.png'
      t_img_file = base_name + 'T_CS.png'
      paths = [file, gt_img_file, s_img_file, t_img_file]
      if 'F' in wb_settings:
        f_img_file = base_name + 'F_CS.png'
        paths.append(f_img_file)
      if 'C' in wb_settings:
        c_img_file = base_name + 'C_CS.png'
        paths.append(c_img_file)

      checks = True
      for curr_path in paths:
        checks = checks & path.exists(curr_path)
      assert checks, 'cannot find WB images match target WB settings'
    return True
