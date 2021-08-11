from scipy.sparse import diags
from scipy.sparse.linalg import cg
import numpy as np




def bistochastize(grid, maxiter=10):
  """Compute diagonal matrices to bistochastize a bilateral grid"""
  m = grid.splat(np.ones(grid.npixels))
  n = np.ones(grid.nvertices)
  for i in range(maxiter):
    n = np.sqrt(n * m / grid.blur(n))
  # Correct m to satisfy the assumption of bistochastization regardless
  # of how many iterations have been run.
  m = n * grid.blur(n)
  Dm = diags(m, 0)
  Dn = diags(n, 0)
  return Dn, Dm


class BilateralSolver(object):
  def __init__(self, grid, params):
    self.grid = grid
    self.params = params
    self.Dn, self.Dm = bistochastize(grid)

  def solve(self, x, w):
    # Check that w is a vector or a nx1 matrix
    if w.ndim == 2:
      assert (w.shape[1] == 1)
    elif w.dim == 1:
      w = w.reshape(w.shape[0], 1)
    A_smooth = (self.Dm - self.Dn.dot(self.grid.blur(self.Dn)))
    w_splat = self.grid.splat(w)
    A_data = diags(w_splat[:, 0], 0)
    A = self.params["lam"] * A_smooth + A_data
    xw = x * w
    b = self.grid.splat(xw)
    # Use simple Jacobi preconditioner
    A_diag = np.maximum(A.diagonal(), self.params["A_diag_min"])
    M = diags(1 / A_diag, 0)
    # Flat initialization
    y0 = self.grid.splat(xw) / w_splat
    yhat = np.empty_like(y0)
    for d in range(x.shape[-1]):
      yhat[..., d], info = cg(A, b[..., d], x0=y0[..., d], M=M,
                              maxiter=self.params["cg_maxiter"],
                              tol=self.params["cg_tol"])
    xhat = self.grid.slice(yhat)
    return xhat





