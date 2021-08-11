import seaborn as sns
import os
import bilateral_solver.bilateral_grid as bilateral_grid
import bilateral_solver.bilateral_solver as solver
import numpy as np
import torch
from src import ops

sns.set_style('white')
sns.set_context('notebook')


grid_params = {
    'sigma_luma' : 16, # Brightness bandwidth
    'sigma_chroma': 8, # Color bandwidth
    'sigma_spatial': 16 # Spatial bandwidth
}

bs_params = {
    'lam': 128, # The strength of the smoothness parameter
    'A_diag_min': 1e-5, # Clamp the diagonal of the A diagonal in the Jacobi preconditioner.
    'cg_tol': 1e-5,  # The tolerance on the convergence in PCG
    'cg_maxiter': 25 # The number of PCG iterations
}



def process_image(reference, target, confidence=None, tensor=False):
    if confidence is None:
        confidence = ops.imread(os.path.join('bilateral_solver',
                                           'confidence.png'), gray=True)

    if tensor:
        gpu = reference.is_cuda
        if gpu:
            reference = reference.cpu().data.numpy()
            target = target.cpu().data.numpy()
            reference = reference.transpose((1, 2, 0))
        else:
            reference = reference.data.numpy()
            target = target.data.numpy()
            reference = reference.transpose((1, 2, 0))

    im_shape = reference.shape[:2]
    assert(im_shape[0] == target.shape[0])
    assert(im_shape[1] == target.shape[1])

    confidence = ops.imresize.imresize(confidence, output_shape=im_shape)

    assert(im_shape[0] == confidence.shape[0])
    assert(im_shape[1] == confidence.shape[1])

    grid = bilateral_grid.BilateralGrid(reference, **grid_params)

    t = target.reshape(-1, 1).astype(np.double)
    c = confidence.reshape(-1, 1).astype(np.double)
    output = solver.BilateralSolver(grid,
                                    bs_params).solve(t, c).reshape(im_shape)
    if tensor:
        if gpu:
            output = torch.from_numpy(output).to(
                device=torch.cuda.current_device())
        else:
            output = torch.from_numpy(output)

    return output

