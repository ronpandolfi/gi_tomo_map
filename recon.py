import numpy as np
from alive_progress import alive_it


def sirt(sinogram, projection_operator, num_iterations=10, inverse_operator=None, show_progress=True):
    R = np.diag(1 / np.sum(projection_operator, axis=1, dtype=np.float32))
    R = np.nan_to_num(R)
    C = np.diag(1 / np.sum(projection_operator, axis=0, dtype=np.float32))
    C = np.nan_to_num(C)

    x_rec = np.zeros(projection_operator.shape[1], dtype=np.float32)

    for _ in alive_it(range(num_iterations), title='Calculating reconstruction with SIRT', disable=not show_progress):
        if inverse_operator:
            x_rec += C @ (inverse_operator @ (R @ (sinogram.ravel() - projection_operator @ x_rec)))
        else:
            x_rec += C @ (projection_operator.T @ (R @ (sinogram.ravel() - projection_operator @ x_rec)))

    return x_rec.reshape(projection_operator.shape[2:])