import argparse, json, numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.spatial.transform import Rotation as R

def as_rotation_mats(entry):
    """
    Prefer 'rotation_matrix' if present (shape: (N,3,3)).
    Otherwise fall back to Euler (radians) in 'roll','pitch','yaw' (assumed XYZ).
    """
    if "rotation_matrix" in entry and entry["rotation_matrix"] is not None:
        M = np.array(entry["rotation_matrix"], dtype=float)
        # Ensure shape is (N,3,3)
        if M.ndim == 3 and M.shape[1:] == (3,3):
            return M
        else:
            raise ValueError(f"rotation_matrix has shape {M.shape}, expected (N,3,3)")
    # Fallback to Euler (assumed radians)
    for k in ("roll","pitch","yaw"):
        if k not in entry or entry[k] is None:
            raise ValueError("No rotation_matrix and missing Euler arrays.")
    eul = np.column_stack([entry["roll"], entry["pitch"], entry["yaw"]]).astype(float)
    return R.from_euler("xyz", eul, degrees=False).as_matrix()

def geodesic_step_angles_deg(Rs):
    """
    Geodesic step angles: theta_i = ||Log(R_i^T R_{i+1})||, returned in degrees.
    Rs: (N,3,3)
    """
    rot = R.from_matrix(Rs)
    dR = rot[:-1].inv() * rot[1:]
    # rotation vectors (rad) -> their magnitudes are the angles
    thetas_rad = np.linalg.norm(dR.as_rotvec(), axis=1)
    return np.degrees(thetas_rad)

def downsample_every_20(M, times=None):
    """Take every 20th sample: indices [0,20,40,...]."""
    if times is not None:
        M = M[np.arange(0, len(times), 20)]
    else:
        M = M[::20]
    return M

def omega_step_expect_deg(entry, dt):
    """
    Optional sanity check: estimate per-step angle from |ω| * dt (rad) -> deg.
    Returns None if no omega_* present.
    """
    keys = ("omega_x","omega_y","omega_z")
    if not all(k in entry and entry[k] is not None for k in keys):
        return None
    wx = np.asarray(entry["omega_x"], dtype=float)
    wy = np.asarray(entry["omega_y"], dtype=float)
    wz = np.asarray(entry["omega_z"], dtype=float)
    mag = np.sqrt(wx**2 + wy**2 + wz**2)  # rad/s
    # Downsample the *positions* at 50 Hz; ω is at 1000 Hz—estimate the
    # expected step over 20 ms as mean(|ω|) over each 20-sample block times dt
    if len(mag) < 21:
        return None
    # block means over non-overlapping chunks of length 20
    nblocks = len(mag)//20
    mag_blocks = mag[:nblocks*20].reshape(nblocks, 20).mean(axis=1)
    step_rad = mag_blocks * dt  # rad
    return np.degrees(step_rad)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Path to EXP jsonl file")
    ap.add_argument("--target_hz", type=float, default=50.0)
    ap.add_argument("--top", type=int, default=10, help="Show top-N largest steps")
    args = ap.parse_args()

    path = Path(args.jsonl)
    if not path.exists():
        raise FileNotFoundError(path)

    dt = 1.0 / args.target_hz

    all_steps = []
    per_impact = []

    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            # Only process EXP GT records
            if entry.get("Data source") != "EXP":
                continue

            # Build rotation sequence from GT (preferring rotation_matrix)
            M = as_rotation_mats(entry)    # (N,3,3)
            times = np.asarray(entry["times"], dtype=float) if "times" in entry else None

            # Downsample to ~50 Hz by taking every 20th sample
            M50 = downsample_every_20(M, times=times)

            # Compute per-step geodesic angles at 50 Hz
            theta_deg = geodesic_step_angles_deg(M50)
            all_steps.append(theta_deg)

            # Optional sanity: expected per-step angle from |ω| * Δt
            exp_deg = omega_step_expect_deg(entry, dt)

            per_impact.append({
                "id": entry.get("Impact ID"),
                "participant": entry.get("Participant"),
                "type": entry.get("Type"),
                "repeat": entry.get("Repeat"),
                "n_steps": len(theta_deg),
                "theta_deg": theta_deg,
                "omega_est_deg": exp_deg,  # may be None
            })

    if not all_steps:
        print("No EXP records found.")
        return

    steps = np.concatenate(all_steps)
    print(f"\n=== Geodesic step angles at {args.target_hz:.1f} Hz (deg) ===")
    print(f"Impacts used : {len(per_impact)}")
    print(f"Total steps  : {len(steps)}\n")

    def q(p): return np.percentile(steps, p)
    print("Summary (deg per step):")
    print(f"  min/mean/std/max : {steps.min():.3f} / {steps.mean():.3f} / {steps.std():.3f} / {steps.max():.3f}")
    for p in (50, 90, 95, 97.5, 99, 99.5, 99.9):
        print(f"  p{p:<4}: {q(p):.3f}")

    # Show top-N largest steps with their impact IDs for debugging
    print(f"\nTop {args.top} largest steps:")
    # Build a list of (angle, id, idx)
    records = []
    for imp in per_impact:
        for i, val in enumerate(imp["theta_deg"]):
            records.append((val, imp["id"], i, imp["participant"], imp["type"], imp["repeat"]))
    records.sort(reverse=True, key=lambda x: x[0])
    for val, iid, idx, pid, typ, rep in records[:args.top]:
        print(f"  {val:8.3f} deg  | Impact {iid} ({pid} {typ}{rep})  step {idx}->{idx+1}")

    # Quick plausibility check against |ω|*Δt
    have_omega = any(imp["omega_est_deg"] is not None for imp in per_impact)
    if have_omega:
        # pool the expected deg from omega
        est = np.concatenate([imp["omega_est_deg"] for imp in per_impact if imp["omega_est_deg"] is not None])
        print("\nSanity check from |ω|·Δt (deg per 20 ms block):")
        print(f"  mean ≈ {est.mean():.3f},  p95 ≈ {np.percentile(est,95):.3f},  max ≈ {est.max():.3f}")
        print("  (These should be of similar order to the geodesic per-step angles.)")
    else:
        print("\n(No omega_* fields present for |ω|·Δt sanity check.)")

if __name__ == "__main__":
    main()
