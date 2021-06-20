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
from torchvision.utils import save_image


def test_net(net, device, data_dir, model_name, out_dir, batch_size=32):
  """ Tests a trained network and saves the trained model in harddisk.

  Args:
    net: network object (wb_net.WBnet).
    device: use 'cpu' or 'cuda' (string).
    model_name: Name of model.
  """

  input_files = dataset.Data.load_files(data_dir)

  if input_files == []:
    input_files = dataset.Data.load_files(data_dir, mode='testing')

  train_set = dataset.Data(input_files, mode='testing')

  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)


  logging.info(f'''Starting testing:
        Model Name:            {model_name}
        Batch size:            {batch_size}
        Output dir:            {out_dir}
        Device:                {device.type}
  ''')

  if path.exists(out_dir) is not True:
    os.mkdir(out_dir)

  with torch.no_grad():
    for batch in train_loader:
      img = batch['image']
      img = img.to(device=device, dtype=torch.float32)
      result, weights = net(img)

      d_img = batch['fs_d_img']
      d_img = d_img.to(device=device, dtype=torch.float32)
      s_img = batch['fs_s_img']
      s_img = s_img.to(device=device, dtype=torch.float32)
      t_img = batch['fs_t_img']
      t_img = t_img.to(device=device, dtype=torch.float32)
      filename = batch['filename']
      weights = F.interpolate(
        weights, size=(d_img.shape[2], d_img.shape[3]),
        mode='bilinear', align_corners=True)

      out_img = torch.unsqueeze(weights[:, 0, :, :], dim=1) * d_img
      out_img += torch.unsqueeze(weights[:, 1, :, :], dim=1) * s_img
      out_img += torch.unsqueeze(weights[:, 2, :, :], dim=1) * t_img
      for i, fname in enumerate(filename):
        result = ops.to_image(out_img[i, :, :, :])
        fname = path.join(out_dir, path.basename(fname) + '_WB.png')
        result.save(fname)

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
                      default=True,
                      help='Apply BN in network')

  parser.add_argument('-ml', '--model-location', dest='model_location',
                      default=None)

  parser.add_argument('-ted', '--testing-dir', dest='tedir',
                      #default='./data/images/',
                      default='./data/flickr/',
                      #default='./data/cube+/',
                      help='Testing directory')

  parser.add_argument('-od', '--outdir', dest='outdir',
                      default='./results/',
                      help='Results directory')

  parser.add_argument('-g', '--gpu', dest='gpu', default=0, type=int)

  parser.add_argument('-mn', '--model-name', dest='model_name', type=str,
                      default='WB_model_p_64_w_BN_84', help='Model name')

  return parser.parse_args()

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
  logging.info('Testing Mixed WB correction')
  args = get_args()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if device.type != 'cpu':
    torch.cuda.set_device(args.gpu)

  logging.info(f'Using device {device}')

  net = wb_net.WBnet(device=device, norm=args.norm)

  model_path = os.path.join('models', args.model_name + '.pth')

  net.load_state_dict(torch.load(model_path, map_location=device))

  logging.info(f'Model loaded from {model_path}')

  net.to(device=device)

  net.eval()

  test_net(net=net, device=device, data_dir=args.tedir,
           batch_size=args.batch_size, out_dir=args.outdir,
           model_name=args.model_name)
