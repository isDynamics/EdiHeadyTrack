# EdiHeadyTrack/filter.py
import numpy as np
from typing import Iterable, Literal, Optional, Dict, Any
from scipy.spatial.transform import Rotation as R

class SO3Filter:
    """
    Geodesic Gaussian smoother on SO(3).

    This filter removes frame-to-frame jitter in rotation sequences while
    preserving valid rotations. It performs Gaussian-weighted smoothing
    directly on the manifold using the log-exp maps.

    Based on:
        Lee & Shin (2001). "General Construction of Time-Domain Filters
        for Orientation Data." IEEE Trans. Visualization & Computer Graphics.

    Parameters
    ----------
    sigma : float, default=1.0
        Standard deviation of the Gaussian kernel (in frames).
    window : int, optional
        Window radius (in frames). Default = 3 * sigma.
    spike_deg : float or None, default=20
        Detects and clamps single-frame spikes larger than this threshold.
        Set to None to disable.
    sample_rate_hz : float, default=50.0
        Sampling rate (used to compute angular velocity).
    compute_omega : bool, default=True
        If True, returns angular velocity (rad/s).

    Usage
    -----
    >>> filt = SO3Filter(sigma=1.5)
    >>> out = filt.smooth(rot_matrices, input_type="matrix")
    >>> out["R_smooth"], out["quat"], out["omega"]
    """

    def __init__(
        self,
        sigma: float = 1.0,
        window: Optional[int] = None,
        spike_deg: Optional[float] = 20.0,
        sample_rate_hz: float = 50.0,
        compute_omega: bool = True,
    ):
        self.sigma = sigma
        self.window = int(3 * sigma) if window is None else window
        self.spike_deg = spike_deg
        self.sample_rate_hz = sample_rate_hz
        self.compute_omega = compute_omega

    # ---------- Public API ----------

    def smooth(
        self,
        rotations: Iterable,
        input_type: Literal["matrix", "quat"] = "matrix"
    ) -> Dict[str, Any]:
        Rs = self._to_rotation_sequence(rotations, input_type)

        if self.spike_deg is not None and len(Rs) >= 3:
            Rs = self._spike_clamp(Rs, np.deg2rad(self.spike_deg))

        R_smooth = self._gaussian_smooth(Rs, self.sigma, self.window)

        out = {
            "R_smooth": R_smooth.as_matrix(),
            "quat": R_smooth.as_quat(),  # (x,y,z,w)
        }

        if self.compute_omega:
            out["omega"] = self._angular_velocity(R_smooth, 1.0 / self.sample_rate_hz)

        return out

    # ---------- Internal helpers ----------

    @staticmethod
    def _to_rotation_sequence(rotations: Iterable, input_type: str) -> R:
        arr = np.asarray(rotations)
        if input_type == "matrix":
            if arr.ndim != 3 or arr.shape[1:] != (3, 3):
                raise ValueError("Rotation matrices must have shape (N,3,3).")
            return R.from_matrix(arr)
        elif input_type == "quat":
            if arr.ndim != 2 or arr.shape[1] != 4:
                raise ValueError("Quaternions must have shape (N,4) [x,y,z,w].")
            # Hemisphere fix
            q = arr.copy()
            for i in range(1, len(q)):
                if np.dot(q[i-1], q[i]) < 0:
                    q[i] = -q[i]
            return R.from_quat(q)
        else:
            raise ValueError("input_type must be 'matrix' or 'quat'.")

    @staticmethod
    def _geo_midpoint(Ra: R, Rb: R) -> R:
        d = (Ra.inv() * Rb).as_rotvec()
        return Ra * R.from_rotvec(0.5 * d)

    def _spike_clamp(self, Rs: R, thresh_rad: float) -> R:
        """Replace single-frame spikes with geodesic midpoint of neighbors."""
        seq = list(Rs)
        for i in range(1, len(seq) - 1):
            a = (seq[i-1].inv() * seq[i]).magnitude()
            b = (seq[i].inv()   * seq[i+1]).magnitude()
            c = (seq[i-1].inv() * seq[i+1]).magnitude()
            if (a + b) - c > thresh_rad:
                seq[i] = self._geo_midpoint(seq[i-1], seq[i+1])
        return R.concatenate(seq)

    def _gaussian_kernel(self, sigma: float, window: int) -> np.ndarray:
        k = np.arange(-window, window + 1)
        w = np.exp(-(k**2) / (2 * sigma**2))
        return w / np.sum(w)

    def _gaussian_smooth(self, Rs: R, sigma: float, window: int) -> R:
        """Geodesic Gaussian smoothing in log space (Lee & Shin 2001)."""
        N = len(Rs)
        kernel = self._gaussian_kernel(sigma, window)
        seq = []

        for i in range(N):
            v = np.zeros(3)
            wsum = 0.0
            R_ref = Rs[i]

            for k, kw in zip(range(-window, window + 1), kernel):
                j = i + k
                if 0 <= j < N:
                    dR = R_ref.inv() * Rs[j]
                    v += kw * dR.as_rotvec()
                    wsum += kw

            seq.append(R_ref * R.from_rotvec(v / wsum))

        return R.concatenate(seq)

    @staticmethod
    def _angular_velocity(Rs: R, dt: float) -> np.ndarray:
        """Angular velocity ω = Log(Rᵢᵀ Rᵢ₊₁)/dt, rad/s."""
        N = len(Rs)
        if N <= 1:
            return np.zeros((N, 3))
        w = np.zeros((N, 3))
        for i in range(N - 1):
            dR = Rs[i].inv() * Rs[i + 1]
            w[i] = dR.as_rotvec() / dt
        w[-1] = w[-2]
        return w

