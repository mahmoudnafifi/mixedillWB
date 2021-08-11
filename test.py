import argparse
import logging
import torch
from src import wb_net
import os.path as path
import os
from src import ops
from src import dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from src import weight_refinement as weight_refinement




def test_net(net, device, data_dir, model_name, out_dir, save_weights,
             multi_scale=False, keep_aspect_ratio=False, t_size=128,
             post_process=False, batch_size=32, wb_settings=None):
  """ Tests a trained network and saves the trained model in harddisk.
  """
  if wb_settings is None:
    wb_settings = ['D', 'S', 'T', 'F', 'C']
  input_files = dataset.Data.load_files(data_dir)

  if input_files == []:
    input_files = dataset.Data.load_files(data_dir, mode='testing')

  if multi_scale:
    test_set = dataset.Data(input_files, mode='testing', t_size=t_size,
                            wb_settings=wb_settings,
                            keep_aspect_ratio=keep_aspect_ratio)
  else:
    test_set = dataset.Data(input_files, mode='testing', t_size=t_size,
                            wb_settings=wb_settings,
                            keep_aspect_ratio=keep_aspect_ratio)

  test_set = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)


  logging.info(f'''Starting testing:
        Model Name:            {model_name}
        Batch size:            {batch_size}
        Output dir:            {out_dir}
        WB settings:           {wb_settings}
        Save weights:          {save_weights}
        Device:                {device.type}
  ''')

  if path.exists(out_dir) is not True:
    os.mkdir(out_dir)

  with torch.no_grad():

    for batch in test_set:

      img = batch['image']

      img = img.to(device=device, dtype=torch.float32)
      _, weights = net(img)
      if multi_scale:
        img_1 = F.interpolate(
          img, size=(int(0.5 * img.shape[2]), int(0.5 * img.shape[3])),
          mode='bilinear', align_corners=True)
        _, weights_1 = net(img_1)
        weights_1 = F.interpolate(weights_1, size=(img.shape[2], img.shape[3]),
                                 mode='bilinear', align_corners=True)
        img_2 = F.interpolate(
          img, size=(int(0.25 * img.shape[2]), int(0.25 * img.shape[3])),
          mode='bilinear', align_corners=True)
        _, weights_2 = net(img_2)
        weights_2 = F.interpolate(weights_2, size=(img.shape[2], img.shape[3]),
                                 mode='bilinear', align_corners=True)
        weights = (weights + weights_1 + weights_2) / 3

      d_img = batch['fs_d_img']
      d_img = d_img.to(device=device, dtype=torch.float32)
      s_img = batch['fs_s_img']
      s_img = s_img.to(device=device, dtype=torch.float32)
      t_img = batch['fs_t_img']
      t_img = t_img.to(device=device, dtype=torch.float32)
      imgs = [d_img, s_img, t_img]
      if 'F' in wb_settings:
        f_img = batch['fs_f_img']
        f_img = f_img.to(device=device, dtype=torch.float32)
        imgs.append(f_img)
      if 'C' in wb_settings:
        c_img = batch['fs_c_img']
        c_img = c_img.to(device=device, dtype=torch.float32)
        imgs.append(c_img)

      filename = batch['filename']
      weights = F.interpolate(
        weights, size=(d_img.shape[2], d_img.shape[3]),
        mode='bilinear', align_corners=True)

      if post_process:
        for i in range(weights.shape[1]):
          for j in range(weights.shape[0]):
            ref = imgs[0][j, :, :, :]
            curr_weight = weights[j, i, :, :]
            refined_weight = weight_refinement.process_image(ref, curr_weight,
                                                             tensor=True)
            weights[j, i, :, :] = refined_weight
            weights = weights / torch.sum(weights, dim=1)


      for i in range(weights.shape[1]):
        if i == 0:
          out_img = torch.unsqueeze(weights[:, i, :, :], dim=1) * imgs[i]
        else:
          out_img += torch.unsqueeze(weights[:, i, :, :], dim=1) * imgs[i]

      for i, fname in enumerate(filename):
        result = ops.to_image(out_img[i, :, :, :])
        name = path.join(out_dir, path.basename(fname) + '_WB.png')
        result.save(name)
        if save_weights:
          # save weights
          postfix = ['D', 'S', 'T']
          if 'F' in wb_settings:
            postfix.append('F')
          if 'C' in wb_settings:
            postfix.append('C')
          for j in range(weights.shape[1]):
            weight = torch.tile(weights[:, j, :, :], dims=(3, 1, 1))
            weight = ops.to_image(weight)
            name = path.join(out_dir, path.basename(fname) +
                             f'_weight_{postfix[j]}.png')
            weight.save(name)


  logging.info('End of testing')



def get_args():
  """ Gets command-line arguments.

  Returns:
    Return command-line arguments as a set of attributes.
  """

  parser = argparse.ArgumentParser(description='Test WB Correction.')

  parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?',
                      default=1, help='Batch size', dest='batch_size')

  parser.add_argument('-nrm', '--normalization', dest='norm', type=bool,
                      default=False,
                      help='Apply BN in network')

  parser.add_argument('-ml', '--model-location', dest='model_location',
                      default=None)

  parser.add_argument('-wbs', '--wb-settings', dest='wb_settings', nargs='+',
                      default=['D', 'S', 'T'])
                      # default=['D', 'S', 'T', 'F', 'C'])

  parser.add_argument('-sw', '--save-weights', dest='save_weights',
                      default=True, type=bool)

  parser.add_argument('-ka', '--keep-aspect-ratio', dest='keep_aspect_ratio',
                      default=False, type=bool,
                      help='To keep aspect ratio before processing. Only '
                           'works when multi-scale is off.')

  parser.add_argument('-ms', '--multi-scale', dest='multi_scale',
                      default=True, type=bool)

  parser.add_argument('-pp', '--post-process', dest='post_process',
                      default=True, type=bool)

  parser.add_argument('-ted', '--testing-dir', dest='tedir',
                      default='./data/images/',
                      help='Testing directory')

  parser.add_argument('-od', '--outdir', dest='outdir',
                      default='./results/',
                      help='Results directory')

  parser.add_argument('-g', '--gpu', dest='gpu', default=0, type=int)

  parser.add_argument('-ts', '--target-size', dest='t_size', default=384,
                      type=int,
                      help='Size before feeding images to the network. '
                           'Typically, 128 or 256 give good results. If '
                           'multi-scale is used, then 384 is recommended.')

  parser.add_argument('-mn', '--model-name', dest='model_name', type=str,
                      default='WB_model_p_64_D_S_T',
                      #default='WB_model_p_64_D_S_T',
                      help='Model name')

  return parser.parse_args()

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
  logging.info('Testing Mixed-Ill WB correction')
  args = get_args()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if device.type != 'cpu':
    torch.cuda.set_device(args.gpu)


  logging.info(f'Using device {device}')

  net = wb_net.WBnet(device=device, norm=args.norm, inchnls=3 * len(
    args.wb_settings))

  model_path = os.path.join('models', args.model_name + '.pth')

  net.load_state_dict(torch.load(model_path, map_location=device))

  logging.info(f'Model loaded from {model_path}')

  net.to(device=device)

  net.eval()

  test_net(net=net, device=device, data_dir=args.tedir,
           batch_size=args.batch_size, out_dir=args.outdir,
           post_process=args.post_process,
           keep_aspect_ratio=args.keep_aspect_ratio,
           t_size=args.t_size,
           multi_scale=args.multi_scale, model_name=args.model_name,
           save_weights=args.save_weights,
           wb_settings=args.wb_settings)
