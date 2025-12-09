import os
import json
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import numpy as np
from sixdrepnet import SixDRepNet
from head_detector import HeadDetector as VGGHeadDetector
import gc
import torch


class HeadPoseModel:
    def __init__(self, model_name='6drepnet'):
        """
        Initialize a head pose estimation model.
        Currently, only '6drepnet' is supported.
        """
        self.model_name = model_name
        if model_name == '6drepnet':        
            self.model = SixDRepNet(model_type='SixDRepNet') 
        elif model_name == '6drepnet360':
            self.model = SixDRepNet(model_type='SixDRepNet360')
        elif model_name == 'vggheads':
            return
        else:
            raise ValueError("Currently, only '6drepnet' is implemented.")

    def predict(self, head_images, camera=None, viz_output=None, data_output=None, 
                use_jsonl=False, trial_metadata=None, sample_rate=50.0):
        """
        Run head pose estimation on cropped head images.
        :param head_images: List of cropped head images (from HeadDetector)
        :param camera: Camera object with extrinsic parameters (optional)
        :param viz_output: Directory to save visualized output
        :param data_output: Directory to save pose data (or JSONL file path if use_jsonl=True)
        :param use_jsonl: If True, save data in JSONL format instead of individual .npy files
        :param trial_metadata: Dict with metadata like {'participant': 'P003024', 'trial': 'L1_F', 'impact_id': 1}
        :param sample_rate: Sample rate in Hz for time array generation
        :return: Tuple containing:
                 - List of local rotation matrices
                 - List of global rotation matrices
                 - List of local Euler angles
                 - List of global Euler angles
        """
        euler_locals = []
        euler_globals = []
        rmat_locals = []
        rmat_globals = []
        quaternions = []
        rotation_matrices = []

        # Create directories for legacy format
        if not use_jsonl:
            os.makedirs(viz_output, exist_ok=True) if viz_output else None
            os.makedirs(data_output, exist_ok=True) if data_output else None

        for i, img in enumerate(tqdm(head_images, desc=f"Estimating head pose for {os.path.basename(data_output) if data_output else 'video'}")):
            
            if self.model_name == '6drepnet' or self.model_name == '6drepnet360':
                # Infer euler angles from the model
                pitch, yaw, roll = self.model.predict(img)
                euler_local = [pitch[0], yaw[0], roll[0]]
            elif self.model_name == 'vggheads':
                # Reinitialize model each frame to avoid internal state buildup / slowdown
                self.vgg_head_model = VGGHeadDetector()

                # Force cleanup before processing
                gc.collect()
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except ImportError:
                    pass

                # Ensure frame is uint8 and contiguous (VGGHeads expects BGR uint8 like OpenCV)
                if img.dtype != np.uint8:
                    frame = img.astype(np.uint8)
                else:
                    frame = img
                frame = np.ascontiguousarray(frame)

                # Run prediction
                predictions = self.vgg_head_model(frame)

                # If no heads detected, fallback to zero angles to keep pipeline stable
                if not hasattr(predictions, 'heads') or not predictions.heads:
                    euler_local = [0.0, 0.0, 0.0]
                else:
                    head = predictions.heads[0]
                    pose = head.head_pose
                    roll = float(pose.roll)
                    pitch = float(pose.pitch)
                    yaw = float(pose.yaw)
                    euler_local = [pitch, yaw, roll]

                # Force GPU memory cleanup after processing to prevent accumulation
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except ImportError:
                    pass

            rot_local = R.from_euler('yzx', euler_local, degrees=True)
            euler_local = rot_local.as_euler('xyz', degrees=False)
            # print(pitch)
            euler_local = [-angle for angle in euler_local]  # Match MoCap convention
            rot_local = R.from_euler('xyz', euler_local, degrees=False)
            rmat_local = rot_local.as_matrix()

            euler_locals.append(euler_local)
            rmat_locals.append(rmat_local)

            # Store quaternion and rotation matrix for JSONL format
            quaternion = rot_local.as_quat()  # [x, y, z, w] format
            quaternions.append([quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
            rotation_matrices.append(rmat_local.tolist())

            # Save individual .npy files for legacy format
            if data_output and not use_jsonl:
                np.save(
                    os.path.join(data_output, f"{i:04d}.npy"),
                    np.array(euler_local)
                )

            # Apply camera extrinsics (if any)
            if camera:
                camera_rot = R.from_matrix(camera.get_rotation_matrix())
                rot_global = camera_rot * rot_local
                rmat_global = rot_global.as_matrix()
                rmat_globals.append(rmat_global)
                euler_global = R.from_matrix(rmat_global).as_euler('xyz', degrees=False)
                euler_globals.append(euler_global)
            else:
                rmat_globals.append(rmat_local)
                euler_globals.append(euler_local)

        # Save data in JSONL format if requested
        if use_jsonl and data_output:
            self._save_jsonl_data(
                euler_locals, quaternions, rotation_matrices, 
                data_output, trial_metadata, sample_rate
            )

        return (
            rmat_locals,
            rmat_globals,
            euler_locals,
            euler_globals
        )

    def _save_jsonl_data(self, euler_locals, quaternions, rotation_matrices, 
                        data_output, trial_metadata, sample_rate):
        """
        Save pose data in JSONL format matching dataset.jsonl structure.
        
        Args:
            euler_locals: List of [roll, pitch, yaw] angles for each frame
            quaternions: List of quaternions for each frame
            rotation_matrices: List of rotation matrices for each frame
            data_output: Directory path to save the JSONL file
            trial_metadata: Dictionary containing trial information
            sample_rate: Sample rate for time calculation
        """
        # Create time array based on frame count and sample rate
        num_frames = len(euler_locals)
        times = [i / sample_rate for i in range(num_frames)]
        
        # Extract roll, pitch, yaw arrays
        rolls = [frame[0] for frame in euler_locals]
        pitches = [frame[1] for frame in euler_locals]
        yaws = [frame[2] for frame in euler_locals]
        
        # Create omega arrays (angular velocities) - using simple finite differences
        omega_x = [0.0]  # First frame has zero velocity
        omega_y = [0.0]
        omega_z = [0.0]
        
        dt = 1.0 / sample_rate
        for i in range(1, num_frames):
            # Calculate angular velocities using finite differences
            omega_x.append((rolls[i] - rolls[i-1]) / dt)
            omega_y.append((pitches[i] - pitches[i-1]) / dt)
            omega_z.append((yaws[i] - yaws[i-1]) / dt)
        
        # Create the JSONL entry
        jsonl_entry = {
            "Impact ID": trial_metadata.get("impact_id", "unknown") if trial_metadata else "unknown",
            "Participant": trial_metadata.get("participant", "unknown") if trial_metadata else "unknown",
            "Sample Rate": sample_rate,
            "times": times,
            "omega_x": omega_x,
            "omega_y": omega_y,
            "omega_z": omega_z,
            "roll": rolls,
            "pitch": pitches,
            "yaw": yaws,
            "quaternion": quaternions,
            "rotation_matrix": rotation_matrices
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(data_output, exist_ok=True)
        
        # Save to JSONL file
        jsonl_filename = os.path.join(data_output, "pose_data.jsonl")
        with open(jsonl_filename, 'w') as f:
            f.write(json.dumps(jsonl_entry) + '\n')
        
        print(f"Saved pose data to {jsonl_filename}")

    def compute_relative_rotations(self, rmat_list, data_output=None):
        """
        Compute relative rotations (in degrees):
        - frame-to-frame changes
        - cumulative changes from the first frame
        :param rmat_list: list of 3x3 rotation matrices
        :return: two lists of angles in degrees
        """
        relative_rotations_framewise = [0.0]
        relative_rotations_from_start = [0.0]

        if not rmat_list:
            return relative_rotations_framewise, relative_rotations_from_start

        rmat_first = R.from_matrix(rmat_list[0])

        for i in range(1, len(rmat_list)):
            R_prev = R.from_matrix(rmat_list[i - 1])
            R_curr = R.from_matrix(rmat_list[i])
            R_rel = R_curr * R_prev.inv()
            angle_framewise = np.degrees(R_rel.magnitude())
            relative_rotations_framewise.append(angle_framewise)

            R_cumulative = R_curr * rmat_first.inv()
            angle_cumulative = np.degrees(R_cumulative.magnitude())
            relative_rotations_from_start.append(angle_cumulative)

            if data_output:
                np.savez(
                    os.path.join(data_output, f"{i:04d}_f2f.npz"),
                    euler_local=np.array(relative_rotations_framewise)
                )
                np.savez(
                    os.path.join(data_output, f"{i:04d}_from_start.npz"),
                    euler_local=np.array(relative_rotations_from_start)
                )

        return relative_rotations_framewise, relative_rotations_from_start