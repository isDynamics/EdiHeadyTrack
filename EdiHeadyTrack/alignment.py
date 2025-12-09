# EdiHeadyTrack/so3_align.py
import numpy as np
import cv2

def _project_to_so3(R):
    """Orthonormalize a near-rotation matrix via SVD, enforcing det=+1."""
    U, _, Vt = np.linalg.svd(R)
    R_hat = U @ Vt
    if np.linalg.det(R_hat) < 0:
        U[:, -1] *= -1
        R_hat = U @ Vt
    return R_hat

def _mean_rotation(deltas, tol=1e-4, max_iter=100000):
    """
    Geodesic mean (on SO(3)) of a set of rotation matrices, using
    an iterative log/exp (Rodrigues) update like the reference code.
    Frames with angle > pi/2 are ignored for stability (as in their code).
    """
    # Pre-filter large rotations (angle = ||rvec||)
    keep = []
    for D in deltas:
        rvec, _ = cv2.Rodrigues(D)
        # if np.linalg.norm(rvec) < np.pi / 2:
        keep.append(D)
    # if not keep:
    #     return np.eye(3, dtype=float)

    mats = np.stack(keep)
    mean_R = _project_to_so3(mats[0])
    for _ in range(max_iter):
        # displacement from mean to each sample in Lie algebra
        rvecs = []
        for Di in mats:
            rvec, _ = cv2.Rodrigues(mean_R.T @ Di)
            rvecs.append(rvec.reshape(3))
        disp = np.mean(np.stack(rvecs, axis=0), axis=0)
        d_norm = np.linalg.norm(disp)
        if d_norm < tol:
            break
        dR, _ = cv2.Rodrigues(disp)
        mean_R = _project_to_so3(mean_R @ dR)
    return mean_R

def align_to_gt(pred_mats, gt_mats, mask=None, tol=1e-4, max_iter=100000, return_delta=False):
    """
    Left-align predicted rotations to ground-truth by removing a constant
    'systematic' rotation A: pred ≈ A @ gt  =>  aligned = A^T @ pred.

    Args:
        pred_mats: (N,3,3) predicted rotation matrices
        gt_mats:   (N,3,3) ground-truth rotation matrices
        mask:      indices used to estimate A (None => all)
        tol, max_iter: mean solver params
        return_delta: if True, also return A

    Returns:
        aligned_mats: (N,3,3) aligned predictions
        (optional) A: (3,3) estimated systematic rotation
    """
    P = np.asarray(pred_mats, dtype=float)
    G = np.asarray(gt_mats, dtype=float)
    assert P.shape == G.shape and P.ndim == 3 and P.shape[1:] == (3, 3), "Shapes must be (N,3,3)"

    # Project to SO(3) for safety
    P = np.stack([_project_to_so3(R) for R in P])
    G = np.stack([_project_to_so3(R) for R in G])

    idx = np.arange(P.shape[0]) if mask is None else np.asarray(mask)
    # deltas = pred * gt^T  (if pred ≈ A @ gt, the mean(delta) ≈ A)
    deltas = np.einsum('nij,njk->nik', P[idx], np.transpose(G[idx], (0, 2, 1)))
    A = _mean_rotation(deltas, tol=tol, max_iter=max_iter)

    aligned = np.einsum('ij,njk->nik', A.T, P)  # remove A by left-multiplying A^T
    if return_delta:
        return aligned, A
    return aligned
