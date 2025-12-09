import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D proj)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
from scipy.spatial.transform import Rotation as R
import cv2
from pathlib import Path
from importlib.resources import files, as_file  # Python 3.9+


pkg = __package__ or __name__.split('.')[0]
res = files(pkg).joinpath("ediheadytrack.mplstyle")
if res is not None:
    with as_file(res) as style_path:
        plt.style.use(['tableau-colorblind10', str(style_path)])
        
# Access current style’s color cycle
_CYCLE = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
def _c(i):  # safe color picker
    return _CYCLE[i % len(_CYCLE)] if _CYCLE else None

class Plot:
    @staticmethod
    def plot_euler_time_series(output=None, show=False, **eulers):
        """
        Plot time series of roll, pitch, and yaw angles from sensor and predicted data.
        Usage: Plot.plot_euler_time_series(MoCap=arr_gt, ModelA=arr_pred, ...)
        Each array shape: (T, 3) in radians (roll, pitch, yaw).
        """
        labels = [r'$\phi$', r'$\theta$', r'$\psi$']
        sources = list(eulers.keys())

        fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)

        for i in range(3):
            axes[i].set_title(labels[i])
            for j, (key, signal) in enumerate(eulers.items()):
                
                signal = np.rad2deg(np.asarray(signal))
                time = np.linspace(0, 100, len(signal))
                axes[i].set_xlim([0, 100])

                if key.lower() in ('mocap', 'gt', 'groundtruth', 'ground_truth'):
                    axes[i].plot(time, signal[:, i], label='GT', color=_c(2), linewidth=1, alpha=0.7)
                elif key.lower()=='sixdrepnet':
                    axes[i].scatter(
                        time, signal[:, i],
                        label=f'6DRepNet360',
                        color=_c(1), alpha=0.55)
                else:
                    axes[i].scatter(
                        time, signal[:, i],
                        label=f'{key}',
                        color=_c(1), alpha=0.55)
                
        axes[0].set_ylabel('Euler angle (deg)')
        axes[0].legend()
        axes[1].set_xlabel(r'$t$ (ms)')
        plt.tight_layout()

        if output:
            # If user passed an extension, respect it; otherwise default to PDF (style has savefig.* defaults)
            root, ext = os.path.splitext(output)
            plt.savefig(output if ext else f'{output}.pdf')
        if show:
            plt.show()
        plt.close()

    # TODO change the labelling to use keys of supplied analyser objects?
    @staticmethod
    def plot_geodesic_errors(output=None, show=False, to_identity=None, framewise=None, from_start=None):
        """
        Plot geodesic distance time series (e.g., from PoseAnalyzer).
        :param framewise: list-like of arrays shaped (T,) or nested like [[T_vals]]
        :param from_start: list-like of arrays shaped (T,) or nested like [[T_vals]]
        """
        fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)

        # Left: rotation to identity
        if to_identity:
            for i, data in enumerate(to_identity):
                arr = np.asarray(data[0] if isinstance(data, (list, tuple)) and np.ndim(data) > 1 else data)
                time = np.linspace(0, 100, len(arr))
                if i == 0:
                    axes[0].plot(time, arr, label='GT', color=_c(2), linewidth=1, alpha=0.7)
                else:
                    axes[0].scatter(time, arr, label=f'VGGHeads', color=_c(0), alpha=0.55)
                    
        # Middle: rotation relative to first frame
        if from_start:
            for i, data in enumerate(from_start):
                # tolerate [[...]] or [...]
                arr = np.asarray(data[0] if isinstance(data, (list, tuple)) and np.ndim(data) > 1 else data)
                time = np.linspace(0, 100, len(arr))
                if i == 0:
                    axes[1].plot(time, arr, label='GT', color=_c(2), linewidth=1, alpha=0.7)
                else:
                    axes[1].scatter(time, arr, label=f'VGGHeads', color=_c(0), alpha=0.55)

        # Right: rotation relative to previous frame
        if framewise:
            for i, data in enumerate(framewise):
                arr = np.asarray(data[0] if isinstance(data, (list, tuple)) and np.ndim(data) > 1 else data)
                time = np.linspace(0, 100, len(arr))
                if i == 0:
                    axes[2].plot(time, arr, label='GT', color=_c(2), linewidth=1, alpha=0.7)
                else:
                    axes[2].scatter(time, arr, label=f'VGGHeads', color=_c(0), alpha=0.55)

        axes[0].set_title('GE to Identity')
        axes[1].set_title('GE from First Frame')
        axes[2].set_title('GE from Previous Frame')
        
        axes[0].set_ylabel('GE')
        axes[0].legend()
        axes[1].set_xlabel('Time (ms)')
        plt.tight_layout()

        if output:
            root, ext = os.path.splitext(output)
            plt.savefig(output if ext else f'{output}.pdf')
        if show:
            plt.show()
        plt.close()

    @staticmethod
    def plot_angular_velocity_time_series(output=None, show=False, **angular_velocities):
        """
        Plot time series of angular velocities (roll rate, pitch rate, yaw rate).
        Each input array shape: (T, 3) in rad/s.
        """
        labels = [r'$\omega_x$', r'$\omega_y$', r'$\omega_z$']
        fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)

        for i in range(3):
            axes[i].set_title(labels[i])
            for j, (key, angular_velocity) in enumerate(angular_velocities.items()):
                angular_velocity = np.asarray(angular_velocity)
                time = np.linspace(0, 100, len(angular_velocity))
                axes[i].set_xlim([0, 100])
                axes[i].plot(
                    time, angular_velocity[:, i],
                    label=f'{key}',
                    color=_c(j),
                    marker='o',
                    markersize=3,
                    linestyle='--',
                    linewidth=1
                )

        axes[0].set_ylabel('Angular Velocity (rad/s)')
        axes[0].legend()
        axes[1].set_xlabel('Time (ms)')

        plt.tight_layout()

        if output:
            root, ext = os.path.splitext(output)
            plt.savefig(output if ext else f'{output}.pdf')
        if show:
            plt.show()
        plt.close()

    @staticmethod
    def plot_head_orientation(poses, head_detections, output=None, show=True):
        """
        Plot a sequence of 3D head orientations in a row using an STL model,
        with detected face images as background.
        :param poses: list of (roll, pitch, yaw) in radians
        :param head_detections: list of cropped head images (BGR or RGB)
        """
        num_poses = len(poses)
        fig, axes = plt.subplots(1, num_poses, figsize=(8, 3), subplot_kw={'projection': '3d'})
        if num_poses == 1:
            axes = [axes]

        stl_path = os.path.join(os.path.dirname(__file__), 'utils', 'head.stl')

        # Axis visuals
        axis_points = np.array([
            [0, 0, -0.1],  # Lower the axis slightly
            [1, 0, -0.1],  # X
            [0, 1, -0.1],  # Y
            [0, 0, 0.9],   # Z
        ])
        axis_lines = [(0, 1), (0, 2), (0, 3)]
        # Use named tab colors (not dependent on prop cycle)
        axis_colors = ['tab:red', 'tab:green', 'tab:blue']

        time_index = 0
        for ax, pose, img in zip(axes, poses, head_detections):
            pose = list(pose)
            # pose[1] += 0.3  # camera frame adjustment (preserved)

            # Ensure RGB for display
            if img.ndim == 3 and img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w = img.shape[:2]
            scale_factor = w / 28000.0

            head_mesh = trimesh.load_mesh(stl_path)
            head_mesh.apply_scale(scale_factor)

            zoom = 0.0005
            ax.imshow(img, extent=[-w * zoom, w * zoom, -h * zoom, h * zoom], zorder=-1)

            rotation = R.from_euler('xyz', pose, degrees=False).as_matrix()
            # transformed_vertices = (rotation @ head_mesh.vertices.T).T + np.array([0, 0, 0.1])

            # transformed_mesh = trimesh.Trimesh(vertices=transformed_vertices, faces=head_mesh.faces)

            # collection = Poly3DCollection(
            #     transformed_mesh.vertices[transformed_mesh.faces],
            #     facecolors='white',
            #     linewidths=0.5,
            #     edgecolors='k',
            #     alpha=0.5,
            #     shade=True
            # )
            # ax.add_collection3d(collection)
            # ax.collections[-1].set_edgecolor('none')

            # Draw axes
            transformed_axis_points = (rotation @ axis_points.T).T
            for (start, end), c in zip(axis_lines, axis_colors):
                ax.plot(
                    [transformed_axis_points[start, 0], transformed_axis_points[end, 0]],
                    [transformed_axis_points[start, 1], transformed_axis_points[end, 1]],
                    [transformed_axis_points[start, 2], transformed_axis_points[end, 2]],
                    color=c, linewidth=2
                )

            # Add time text below each image
            # time_index = axes.index(ax)
            ax.text2D(0.5, -0.1, fr'$t={time_index * 20}$ ms', transform=ax.transAxes,
                      ha='center', va='center', fontsize=10)
            time_index += 1
            # 3D pane config
            ax.set_xlim([-0.5, 0.5])
            ax.set_ylim([-0.5, 0.5])
            ax.set_zlim([-0.5, 0.5])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_zlabel("")
            ax.grid(False)
            ax.view_init(elev=0, azim=0)
            ax.set_axis_off()
            ax.set_aspect('equal')

        plt.tight_layout()
        if output:
            root, ext = os.path.splitext(output)
            plt.savefig(output if ext else f'{output}.pdf')
        if show:
            plt.show()
        plt.close()
