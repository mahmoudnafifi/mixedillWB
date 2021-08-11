import argparse
import logging
import os
import sys
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from src import wb_net
import random
from src import ops
import torch.nn.functional as F

try:
  from torch.utils.tensorboard import SummaryWriter

  use_tb = True
except ImportError:
  use_tb = False

from src import dataset
from torch.utils.data import DataLoader


def train_net(net, device, data_dir, val_dir=None, epochs=140,
              batch_size=32, lr=0.001, l2reg=0.00001, grad_clip_value=0,
              chkpoint_period=10, val_freq=1, smooth_weight=0.01,
              multiscale=False, wb_settings=None, shuffle_order=True,
              patch_number=12,  optimizer_algo='Adam', max_tr_files=0,
              max_val_files=0, patch_size=128, model_name='WB_model',
              save_cp=True):
  """ Trains a network and saves the trained model in harddisk.
  """

  dir_checkpoint = 'checkpoints_model/'  # check points directory


  SMOOTHNESS_WEIGHT = smooth_weight


  input_files = dataset.Data.load_files(data_dir)
  random.shuffle(input_files)

  if val_dir is not None:
    val_files = dataset.Data.load_files(val_dir)
    random.shuffle(val_files)
  else:
    val_ind = round(len(input_files) * 0.1)
    val_files = input_files[: val_ind]
    input_files = input_files[val_ind:]


  if max_val_files > 0:
    if max_val_files < len(val_files):
      val_files = val_files[:max_val_files]
  if max_tr_files > 0:
    if max_tr_files < len(input_files):
      input_files = input_files[:max_tr_files]

  dataset.Data.assert_files(input_files, wb_settings=wb_settings)
  dataset.Data.assert_files(val_files, wb_settings=wb_settings)

  train_set = dataset.Data(input_files, patch_size=patch_size,
                           patch_number=patch_number, multiscale=multiscale,
                           shuffle_order=shuffle_order, wb_settings=wb_settings)

  val_set = dataset.Data(val_files, patch_size=patch_size, patch_number=1,
                         shuffle_order=shuffle_order, wb_settings=wb_settings)

  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                            num_workers=6, pin_memory=True)

  val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True,
                          num_workers=6, pin_memory=True)


  if use_tb:  # if TensorBoard is used
    writer = SummaryWriter(log_dir='runs/' + model_name,
                           comment=f'LR_{lr}_BS_{batch_size}')
  else:
    writer = None
  global_step = 0

  logging.info(f'''Starting training:
        Model Name:            {model_name}
        Epochs:                {epochs}
        WB Settings:           {wb_settings}
        Batch size:            {batch_size}
        Patch per image:       {patch_number}
        Patch size:            {patch_size} x {patch_size}
        Learning rate:         {lr}
        L2 reg. weight:        {l2reg}
        Smooth weight:         {smooth_weight}
        Validation Freq.:      {val_freq}
        Grad. clipping:        {grad_clip_value}
        Optimizer:             {optimizer_algo}
        Checkpoints:           {save_cp}
        Device:                {device.type}
        TensorBoard:           {use_tb}
  ''')

  if optimizer_algo == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999),
                           weight_decay=l2reg)


  elif optimizer_algo == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=l2reg)

  else:
    raise NotImplementedError

  x_kernel, y_kernel = ops.get_sobel_kernel(device, chnls=len(wb_settings))

  for epoch in range(epochs):

    net.train()
    epoch_loss = 0
    epoch_smoothness_loss = 0
    epoch_rec_loss = 0


    with tqdm(total=len(train_set), desc=f'Epoch {epoch + 1} / {epochs}',
              unit='img') as pbar:
      for batch in train_loader:
        img = batch['image']
        img = img.to(device=device, dtype=torch.float32)
        gt = batch['gt']
        gt = gt.to(device=device, dtype=torch.float32)
        rec_loss = 0
        smoothness_loss = 0

        for p in range(img.shape[1]):
          patch = img[:, p, :, :, :]
          gt_patch = gt[:, p, :, :, :]
          result, weights = net(patch)
          rec_loss += ops.compute_loss(result, gt_patch)

          smoothness_loss += SMOOTHNESS_WEIGHT * (
              torch.sum(F.conv2d(weights, x_kernel, stride=1) ** 2) +
              torch.sum(F.conv2d(weights, y_kernel, stride=1) ** 2))


        rec_loss = rec_loss / img.shape[1]
        smoothness_loss = smoothness_loss / img.shape[1]
        loss = rec_loss + smoothness_loss

        py_loss = loss.item()
        py_rec_loss = rec_loss.item()

        py_smoothness_loss = smoothness_loss.item()
        epoch_smoothness_loss += py_smoothness_loss

        epoch_rec_loss += py_rec_loss
        epoch_loss += py_loss



        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if grad_clip_value > 0:
          torch.nn.utils.clip_grad_value_(net.parameters(), grad_clip_value)

        if use_tb:
          # for visualization
          vis_weights = (weights - torch.min(weights)) / (
              ops.EPS + torch.max(weights) - torch.min(weights))

          writer.add_scalar('Loss/train', py_loss, global_step)
          writer.add_scalar('Rec Loss/train', py_rec_loss, global_step)
          writer.add_scalar('Smoothness Loss/train', py_smoothness_loss,
                            global_step)

          writer.add_images('Input (1)', patch[:, 0:3, :, :], global_step)
          writer.add_images('Weight (1)',
                            torch.unsqueeze(vis_weights[:, 0, :, :], dim=1),
                            global_step)
          writer.add_images('Input (2)', patch[:, 3:6, :, :], global_step)
          writer.add_images('Weight (2)',
                            torch.unsqueeze(vis_weights[:, 1, :, :], dim=1),
                            global_step)
          writer.add_images('Input (3)', patch[:, 6:9, :, :], global_step)
          writer.add_images('Weight (3)',
                            torch.unsqueeze(vis_weights[:, 2, :, :], dim=1),
                            global_step)
          if vis_weights.shape[1] == 4:
            writer.add_images('Input (4)', patch[:, 9:12, :, :], global_step)
            writer.add_images('Weight (4)',
                              torch.unsqueeze(vis_weights[:, 3, :, :], dim=1),
                              global_step)
          if vis_weights.shape[1] == 5:
            writer.add_images('Input (4)', patch[:, 9:12, :, :], global_step)
            writer.add_images('Weight (4)',
                              torch.unsqueeze(vis_weights[:, 3, :, :], dim=1),
                              global_step)
            writer.add_images('Input (5)', patch[:, 12:, :, :], global_step)
            writer.add_images('Weight (5)',
                              torch.unsqueeze(vis_weights[:, 4, :, :], dim=1),
                              global_step)

          writer.add_images('Result', result, global_step)
          writer.add_images('GT', gt_patch, global_step)

        pbar.update(np.ceil(img.shape[0]))

        pbar.set_postfix(**{'Total loss (batch)': py_loss},
                         **{'Rec. loss (batch)': py_rec_loss},
                         **{'Smoothness loss (batch)': py_smoothness_loss}
                         )

        global_step += 1

    epoch_loss = epoch_loss / (len(train_loader))
    epoch_rec_loss = epoch_rec_loss / (len(train_loader))
    epoch_smoothness_loss = epoch_smoothness_loss / (len(train_loader))
    logging.info(f'{model_name} - Epoch loss: = {epoch_loss}, '
                 f'Rec. loss = {epoch_rec_loss}, '
                 f'Smoothness loss = {epoch_smoothness_loss}')

    if (epoch + 1) % val_freq == 0:
      logging.info('Validation...')
      validation(net=net, loader=val_loader, writer=writer, step=global_step)

    # save a checkpoint
    if save_cp and (epoch + 1) % chkpoint_period == 0:
      if not os.path.exists(dir_checkpoint):
        os.mkdir(dir_checkpoint)
        logging.info('Created checkpoint directory')
      torch.save(net.state_dict(), dir_checkpoint +
                 f'{model_name}_{epoch + 1}.pth')
      logging.info(f'Checkpoint {epoch + 1} saved!')

  # save final trained model
  if not os.path.exists('models'):
    os.mkdir('models')
    logging.info('Created trained models directory')

  torch.save(net.state_dict(), 'models/' + f'{model_name}.pth')
  logging.info('Saved trained model!')

  if use_tb:
    writer.close()

  logging.info('End of training')


def validation(net, loader, writer, step):
  net.eval()
  index = random.randint(0, len(loader) - 1)
  val_loss = 0
  for b, batch in enumerate(loader):
    img = batch['image']
    img = img[:, 0, :, :, :]
    gt = batch['gt']
    gt = gt[:, 0, :, :, :]

    img = img.to(device=device, dtype=torch.float32)
    gt = gt.to(device=device, dtype=torch.float32)

    result, weights = net(img)

    val_loss = ops.compute_loss(result, gt)

    val_loss += val_loss.item()

    if b == index and writer is not None:
      # for visualization
      vis_weights = (weights - torch.min(weights)) / (
          ops.EPS + torch.max(weights) - torch.min(weights))
      writer.add_images('Input (1) [val]', img[:, 0:3, :, :], step)
      writer.add_images('Weight (1) [val]',
                        torch.unsqueeze(vis_weights[:, 0, :, :], dim=1),
                        step)
      writer.add_images('Input (2) [val]', img[:, 3:6, :, :], step)
      writer.add_images('Weight (2) [val]',
                        torch.unsqueeze(vis_weights[:, 1, :, :], dim=1),
                        step)
      writer.add_images('Input (3) [val]', img[:, 6:, :, :], step)
      writer.add_images('Weight (3) [val]',
                        torch.unsqueeze(vis_weights[:, 2, :, :], dim=1),
                        step)

      if vis_weights.shape[1] == 4:
        writer.add_images('Input (4) [val]', img[:, 9:12, :, :], step)
        writer.add_images('Weight (4) [val]',
                          torch.unsqueeze(vis_weights[:, 3, :, :], dim=1),
                          step)
      if vis_weights.shape[1] == 5:
        writer.add_images('Input (4) [val]', img[:, 9:12, :, :], step)
        writer.add_images('Weight (4) [val]',
                          torch.unsqueeze(vis_weights[:, 3, :, :], dim=1),
                          step)
        writer.add_images('Input (5) [val]', img[:, 12:, :, :], step)
        writer.add_images('Weight (5) [val]',
                          torch.unsqueeze(vis_weights[:, 4, :, :], dim=1),
                          step)

      writer.add_images('Result [val]', result, step)
      writer.add_images('GT [val]', gt, step)

  print(f'Validation loss (batch): {val_loss / len(loader)}')
  if writer is not None:
    writer.add_scalar('Validation Loss', val_loss / len(loader), step)

  net.train()


def get_args():
  """ Gets command-line arguments.

  Returns:
    Return command-line arguments as a set of attributes.
  """

  parser = argparse.ArgumentParser(description='Train WB Correction.')
  parser.add_argument('-e', '--epochs', metavar='E', type=int, default=200,
                      help='Number of epochs', dest='epochs')

  parser.add_argument('-s', '--patch-size', dest='patch_size', type=int,
                      default=64, help='Size of input training patches')

  parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?',
                      default=8, help='Batch size', dest='batch_size')

  parser.add_argument('-pn', '--patch-number', type=int, default=4,
                      help='number of patches per trainig image',
                      dest='patch_number')

  parser.add_argument('-opt', '--optimizer', dest='optimizer', type=str,
                      default='Adam', help='Adam or SGD')

  parser.add_argument('-mtf', '--max-tr-files', dest='max_tr_files', type=int,
                      default=0, help='max number of training files; default '
                                      'is 0 which uses all files')

  parser.add_argument('-mvf', '--max-val-files', dest='max_val_files', type=int,
                      default=0, help='max number of validation files; '
                                       'default is 0 which uses all files')

  parser.add_argument('-nrm', '--normalization', dest='norm', type=bool,
                      default=False,
                      help='Apply BN in network')

  parser.add_argument('-msc', '--multi-scale', dest='multiscale', type=bool,
                      default=False,
                      help='Multi-scale training samples')

  parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float,
                      nargs='?', default=1e-4, help='Learning rate', dest='lr')

  parser.add_argument('-l2r', '--l2reg', metavar='L2Reg', type=float,
                      nargs='?', default=0, help='L2 regularization factor',
                      dest='l2r')

  parser.add_argument('-sw', '--smoothness-weight', dest='smoothness_weight',
                      type=float, default=100.0, help='smoothness weight')

  parser.add_argument('-wbs', '--wb-settings', dest='wb_settings', nargs='+',
                      default=['D', 'S', 'T', 'F', 'C'])

  parser.add_argument('-l', '--load', dest='load', type=bool, default=False,
                      help='Load model from a .pth file')

  parser.add_argument('-so', '--shuffle-order', dest='shuffle_order',
                      type=bool, default=False,
                      help='Shuffle order of WB')

  parser.add_argument('-ml', '--model-location', dest='model_location',
                      default=None)

  parser.add_argument('-vf', '--validation-frequency', dest='val_freq',
                      type=int, default=1, help='Validation frequency.')

  parser.add_argument('-cpf', '--checkpoint-frequency', dest='cp_freq',
                      type=int, default=1, help='Checkpoint frequency.')

  parser.add_argument('-gc', '--grad-clip-value', dest='grad_clip_value',
                      type=float, default=0, help='Gradient clipping value; '
                                                  'if = 0, no clipping applied')

  parser.add_argument('-trd', '--training-dir', dest='trdir',
                      default='./data/images/',
                      help='Training directory')

  parser.add_argument('-valdir', '--validation-dir', dest='valdir',
                      default=None, help='Main validation directory')

  parser.add_argument('-g', '--gpu', dest='gpu', default=0, type=int)

  parser.add_argument('-mn', '--model-name', dest='model_name', type=str,
                      default='WB_model', help='Model name')

  return parser.parse_args()

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
  logging.info('Training Mixed-Ill WB correction')
  args = get_args()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if device.type != 'cpu':
    torch.cuda.set_device(args.gpu)

  logging.info(f'Using device {device}')

  net = wb_net.WBnet(device=device, norm=args.norm, inchnls=3 * len(
    args.wb_settings))
  if args.load:
    net.load_state_dict(
      torch.load(args.model_location, map_location=device)
    )
    logging.info(f'Model loaded from {args.model_location}')

  net.to(device=device)


  postfix = f'_p_{args.patch_size}'

  if args.norm:
    postfix += f'_w_BN'

  if args.shuffle_order:
    postfix += f'_w_shuffling'

  if args.smoothness_weight == 0:
    postfix += f'_wo_smoothing'

  for wb_setting in args.wb_settings:
    postfix += f'_{wb_setting}'

  model_name = args.model_name + postfix


  try:
    train_net(net=net, device=device, data_dir=args.trdir,
              patch_number=args.patch_number,
              multiscale=args.multiscale,
              smooth_weight=args.smoothness_weight,
              max_tr_files=args.max_tr_files,
              max_val_files=args.max_val_files,
              wb_settings=args.wb_settings,
              shuffle_order=args.shuffle_order,
              epochs=args.epochs,
              batch_size=args.batch_size, lr=args.lr,
              l2reg=args.l2r,
              optimizer_algo=args.optimizer,
              grad_clip_value=args.grad_clip_value,
              chkpoint_period=args.cp_freq,
              val_freq=args.val_freq, patch_size=args.patch_size,
              model_name=model_name)

  except KeyboardInterrupt:
    torch.save(net.state_dict(), 'wb_correction_intrrupted_check_point.pth')
    logging.info('Saved interrupt checkpoint backup')
    try:
      sys.exit(0)
    except SystemExit:
      os._exit(0)
