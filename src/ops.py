import numpy as np
import torch
from PIL import Image
from src import imresize
from sklearn.linear_model import LinearRegression

mse = torch.nn.MSELoss()


EPS = 1e-9

def get_sobel_kernel(device, chnls=5):
  x_kernel = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
  x_kernel = torch.tensor(x_kernel, dtype=torch.float32).unsqueeze(0).expand(
    1, chnls, 3, 3).to(device=device)
  x_kernel.requires_grad = False
  y_kernel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
  y_kernel = torch.tensor(y_kernel, dtype=torch.float32).unsqueeze(0).expand(
    1, chnls, 3, 3).to(device=device)
  y_kernel.requires_grad = False
  return x_kernel, y_kernel

def compute_loss(source, target):
  return mse(source, target)


def aug(img1, img2, img3, img4, img5=None, img6=None):
  if img5 is not None and img6 is not None:
    assert (img1.shape == img2.shape == img3.shape == img4.shape ==
            img5.shape == img6.shape)
  elif img5 is not None:
    assert (img1.shape == img2.shape == img3.shape == img4.shape ==
            img5.shape)
  else:
    assert img1.shape == img2.shape == img3.shape == img4.shape
  aug_op = np.random.randint(4)
  if aug_op == 3:
    scale = np.random.uniform(low=0.75, high=1.25)
  else:
    scale = 1

  h, w, _ = img1.shape
  if aug_op is 1:
    img1 = np.flipud(img1)
    img2 = np.flipud(img2)
    img3 = np.flipud(img3)
    img4 = np.flipud(img4)
    if img5 is not None:
      img5 = np.flipud(img5)
    if img6 is not None:
      img6 = np.flipud(img6)
  elif aug_op is 2:
    img1 = np.fliplr(img1)
    img2 = np.fliplr(img2)
    img3 = np.fliplr(img3)
    img4 = np.fliplr(img4)
    if img5 is not None:
      img5 = np.fliplr(img5)
    if img6 is not None:
      img6 = np.fliplr(img6)
  elif aug_op is 3:
    img1 = imresize.imresize(img1, scalar_scale=scale)
    img2 = imresize.imresize(img2, scalar_scale=scale)
    img3 = imresize.imresize(img3, scalar_scale=scale)
    img4 = imresize.imresize(img4, scalar_scale=scale)
    if img5 is not None:
      img5 = imresize.imresize(img5, scalar_scale=scale)
    if img6 is not None:
      img6 = imresize.imresize(img6, scalar_scale=scale)
  if img5 is not None and img6 is not None:
    return img1, img2, img3, img4, img5, img6
  elif img5 is not None:
    return img1, img2, img3, img4, img5
  else:
    return img1, img2, img3, img4


def extract_patch(img1, img2, img3, img4, img5=None, img6=None,
                  patch_size=256, patch_number=8):
  if img5 is not None and img6 is not None:
    assert (img1.shape == img2.shape == img3.shape == img4.shape ==
            img5.shape == img6.shape)
  elif img5 is not None:
    assert (img1.shape == img2.shape == img3.shape == img4.shape ==
            img5.shape)
  else:
    assert img1.shape == img2.shape == img3.shape == img4.shape
  h, w, c = img1.shape

  # get random patch coord
  for patch in range(patch_number):
    patch_x = np.random.randint(0, high=w - patch_size)
    patch_y = np.random.randint(0, high=h - patch_size)
    if patch == 0:
      patch1 = np.expand_dims(img1[patch_y:patch_y + patch_size,
                              patch_x:patch_x + patch_size, :], axis=0)

      patch2 = np.expand_dims(img2[patch_y:patch_y + patch_size,
                              patch_x:patch_x + patch_size, :], axis=0)

      patch3 = np.expand_dims(img3[patch_y:patch_y + patch_size,
                              patch_x:patch_x + patch_size, :], axis=0)

      patch4 = np.expand_dims(img4[patch_y:patch_y + patch_size,
                              patch_x:patch_x + patch_size, :], axis=0)
      if img5 is not None:
        patch5 = np.expand_dims(img5[patch_y:patch_y + patch_size,
                                patch_x:patch_x + patch_size, :], axis=0)

      if img6 is not None:
        patch6 = np.expand_dims(img6[patch_y:patch_y + patch_size,
                                patch_x:patch_x + patch_size, :], axis=0)

    else:
      patch1 = np.concatenate((patch1, np.expand_dims(
        img1[patch_y:patch_y + patch_size,
        patch_x:patch_x + patch_size, :], axis=0)), axis=0)

      patch2 = np.concatenate((patch2, np.expand_dims(
        img2[patch_y:patch_y + patch_size,
        patch_x:patch_x + patch_size, :], axis=0)), axis=0)

      patch3 = np.concatenate((patch3, np.expand_dims(
        img3[patch_y:patch_y + patch_size,
        patch_x:patch_x + patch_size, :], axis=0)), axis=0)

      patch4 = np.concatenate((patch4, np.expand_dims(
        img4[patch_y:patch_y + patch_size,
        patch_x:patch_x + patch_size, :], axis=0)), axis=0)

      if img5 is not None:
        patch5 = np.concatenate((patch5, np.expand_dims(
          img5[patch_y:patch_y + patch_size,
          patch_x:patch_x + patch_size, :], axis=0)), axis=0)

      if img6 is not None:
        patch6 = np.concatenate((patch6, np.expand_dims(
          img6[patch_y:patch_y + patch_size,
          patch_x:patch_x + patch_size, :], axis=0)), axis=0)

  if img5 is not None and img6 is not None:
    return patch1, patch2, patch3, patch4, patch5, patch6
  elif img5 is not None:
    return patch1, patch2, patch3, patch4, patch5
  else:
    return patch1, patch2, patch3, patch4


def to_image(image):
    """ converts to PIL image """
    return Image.fromarray((image * 255).astype(np.uint8))


def outOfGamutClipping(I):
  """ Clips out-of-gamut pixels. """
  I[I > 1] = 1  # any pixel is higher than 1, clip it to 1
  I[I < 0] = 0  # any pixel is below 0, clip it to 0
  return I


def kernelP(I):
  """ Kernel function: kernel(r, g, b) -> (r,g,b,rg,rb,gb,r^2,g^2,b^2,rgb,1)
      Ref: Hong, et al., "A study of digital camera colorimetric characterization
       based on polynomial modeling." Color Research & Application, 2001. """
  return (np.transpose(
    (I[:, 0], I[:, 1], I[:, 2], I[:, 0] * I[:, 1], I[:, 0] * I[:, 2],
     I[:, 1] * I[:, 2], I[:, 0] * I[:, 0], I[:, 1] * I[:, 1],
     I[:, 2] * I[:, 2], I[:, 0] * I[:, 1] * I[:, 2],
     np.repeat(1, np.shape(I)[0]))))


def get_mapping_func(image1, image2):
  """ Computes the polynomial mapping """
  image1 = np.reshape(image1, [-1, 3])
  image2 = np.reshape(image2, [-1, 3])
  m = LinearRegression().fit(kernelP(image1), image2)
  return m


def apply_mapping_func(image, m):
  """ Applies the polynomial mapping """
  sz = image.shape
  image = np.reshape(image, [-1, 3])
  result = m.predict(kernelP(image))
  result = np.reshape(result, [sz[0], sz[1], sz[2]])
  return result


def resize_image(im, target_size):
  """ Resizes a given image to a target size.

  Args:
    im: input ndarray image (height x width x channel).
    target_size: target size (list) in the format [target_height, target_width].

  Returns:
    results the resized image (target_height x target_width x channel).
  """

  h, w, c = im.shape
  if h != target_size[1] or w != target_size[0]:
    im = imresize.imresize(im, output_shape=(target_size[0], target_size[1]))
  if c == 1:
    im = np.expand_dims(im, axis=-1)
  return im


def to_tensor(im, dims=3):
  """ Converts a given ndarray image to torch tensor image.

  Args:
    im: ndarray image (height x width x channel x [sample]).
    dims: dimension number of the given image. If dims = 3, the image should
      be in (height x width x channel) format; while if dims = 4, the image
      should be in (height x width x channel x sample) format; default is 3.

  Returns:
    torch tensor in the format (channel x height x width)  or (sample x
      channel x height x width).
  """

  assert (dims == 3 or dims == 4)
  if dims == 3:
    im = im.transpose((2, 0, 1))
  elif dims == 4:
    im = im.transpose((0, 3, 1, 2))
  else:
    raise NotImplementedError

  return torch.from_numpy(im.copy())

  #if dims == 4:
  #  return torch.from_numpy(np.flip(im, axis=0).copy())
  #else:
  #  return torch.from_numpy(im.copy())



def get_basename(filename):
  parts = filename.split('_')
  base_name = ''
  for i in range(len(parts) - 2):
    base_name = base_name + parts[i] + '_'

  return base_name

def to_image(image):
  """ converts to PIL image """
  image = from_tensor_to_image(image)
  return Image.fromarray((image * 255).astype(np.uint8))

def from_tensor_to_image(tensor):
  """ Converts torch tensor image to numpy tensor image.

  Args:
    tensor: torch image tensor in one of the following formats:
      - 1 x channel x height x width
      - channel x height x width

  Returns:
    return a cpu numpy tensor image in one of the following formats:
      - 1 x height x width x channel
      - height x width x channel
  """

  image = tensor.cpu().numpy()
  if len(image.shape) == 4:
    image = image.transpose(0, 2, 3, 1)
  if len(image.shape) == 3:
    image = image.transpose(1, 2, 0)
  return image


def imread(file, gray=False):
  image = Image.open(file)
  image = np.array(image)
  if not gray:
    image = image[:, :, :3]
  image = im2double(image)
  return image


def aspect_ratio_imresize(im, max_output=256):
  h, w, c = im.shape
  if max(h, w) > max_output:
    ratio = max_output / max(h, w)
    im = imresize.imresize(im, scalar_scale=ratio)
    h, w, c = im.shape

  if w % (2 ** 4) == 0:
    new_size_w = w
  else:
    new_size_w = w + (2 ** 4) - w % (2 ** 4)

  if h % (2 ** 4) == 0:
    new_size_h = h
  else:
    new_size_h = h + (2 ** 4) - h % (2 ** 4)

  new_size = (new_size_h, new_size_w)
  if not ((h, w) == new_size):
    im = imresize.imresize(im, output_shape=new_size)

  return im


def im2double(im):
  """ Converts an uint image to floating-point format [0-1].

  Args:
    im: image (uint ndarray); supported input formats are: uint8 or uint16.

  Returns:
    input image in floating-point format [0-1].
  """

  if im[0].dtype == 'uint8' or im[0].dtype == 'int16':
    max_value = 255
  elif im[0].dtype == 'uint16' or im[0].dtype == 'int32':
    max_value = 65535
  return im.astype('float') / max_value
