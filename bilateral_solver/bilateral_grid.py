import numpy as np
import bilateral_solver.color_transformations as colors

MAX_VAL = 255.0
from scipy.sparse import csr_matrix


def get_valid_idx(valid, candidates):
  """Find which values are present in a list and where they are located"""
  locs = np.searchsorted(valid, candidates)
  # Handle edge case where the candidate is larger than all valid values
  locs = np.clip(locs, 0, len(valid) - 1)
  # Identify which values are actually present
  valid_idx = np.flatnonzero(valid[locs] == candidates)
  locs = locs[valid_idx]
  return valid_idx, locs


class BilateralGrid(object):
  def __init__(self, im, sigma_spatial=32, sigma_luma=8, sigma_chroma=8):
    im_yuv = colors.rgb2yuv(im)
    # Compute 5-dimensional XYLUV bilateral-space coordinates
    Iy, Ix = np.mgrid[:im.shape[0], :im.shape[1]]
    x_coords = (Ix / sigma_spatial).astype(int)
    y_coords = (Iy / sigma_spatial).astype(int)
    luma_coords = (im_yuv[..., 0] / sigma_luma).astype(int)
    chroma_coords = (im_yuv[..., 1:] / sigma_chroma).astype(int)
    coords = np.dstack((x_coords, y_coords, luma_coords, chroma_coords))
    coords_flat = coords.reshape(-1, coords.shape[-1])
    self.npixels, self.dim = coords_flat.shape
    # Hacky "hash vector" for coordinates,
    # Requires all scaled coordinates be < MAX_VAL
    self.hash_vec = (MAX_VAL ** np.arange(self.dim))
    # Construct S and B matrix
    self._compute_factorization(coords_flat)

  def _compute_factorization(self, coords_flat):
    # Hash each coordinate in grid to a unique value
    hashed_coords = self._hash_coords(coords_flat)
    unique_hashes, unique_idx, idx = \
      np.unique(hashed_coords, return_index=True, return_inverse=True)
    # Identify unique set of vertices
    unique_coords = coords_flat[unique_idx]
    self.nvertices = len(unique_coords)
    # Construct sparse splat matrix that maps from pixels to vertices
    self.S = csr_matrix((np.ones(self.npixels), (idx, np.arange(self.npixels))))
    # Construct sparse blur matrices.
    # Note that these represent [1 0 1] blurs, excluding the central element
    self.blurs = []
    for d in range(self.dim):
      blur = 0.0
      for offset in (-1, 1):
        offset_vec = np.zeros((1, self.dim))
        offset_vec[:, d] = offset
        neighbor_hash = self._hash_coords(unique_coords + offset_vec)
        valid_coord, idx = get_valid_idx(unique_hashes, neighbor_hash)
        blur = blur + csr_matrix((np.ones((len(valid_coord),)),
                                  (valid_coord, idx)),
                                 shape=(self.nvertices, self.nvertices))
      self.blurs.append(blur)

  def _hash_coords(self, coord):
    """Hacky function to turn a coordinate into a unique value"""
    return np.dot(coord.reshape(-1, self.dim), self.hash_vec)

  def splat(self, x):
    return self.S.dot(x)

  def slice(self, y):
    return self.S.T.dot(y)

  def blur(self, x):
    """Blur a bilateral-space vector with a 1 2 1 kernel in each dimension"""
    assert x.shape[0] == self.nvertices
    out = 2 * self.dim * x
    for blur in self.blurs:
      out = out + blur.dot(x)
    return out

  def filter(self, x):
    """Apply bilateral filter to an input x"""
    return self.slice(self.blur(self.splat(x))) / \
           self.slice(self.blur(self.splat(np.ones_like(x))))