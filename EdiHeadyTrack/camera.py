import numpy as np
import json
import os

class Camera:
    """
    A class for storing extrinsic camera parameters (and optionally intrinsic parameters).
    Extrinsic parameters can be provided as a dictionary or as a path to a JSON file.
    """
    def __init__(self, extrinsics=str):
        """
        :param extrinsics: Either a dictionary of extrinsic parameters (with keys 'rotation' and 'translation')
                           or a path (string) to a JSON file containing the extrinsics.
                           If not provided, defaults to an identity rotation and zero translation.
        """
        if isinstance(extrinsics, str):
            # If extrinsics is a file path, parse the file.
            self.extrinsics = self._parse_extrinsics_file(extrinsics)
        else:
            raise ValueError("Extrinsic parameters must be provided as a valid file path.")
        
    def _parse_extrinsics_file(self, filepath):
        """
        Parse a JSON file containing camera extrinsic parameters.
        Returns a dictionary with keys:
          - 'rotation': 3x3 numpy array,
          - 'translation': 3-element numpy array.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Error parsing JSON file {filepath}: {e}")

        # Extract translation vector
        translation = np.array([
            data["Translation"]["x"],
            data["Translation"]["y"],
            data["Translation"]["z"]
        ])

        # Extract rotation matrix
        rotation = np.array(data["Rotation Matrix"]).T

        permutation = np.array([[0, 1, 0],
                                [0, 0, 1],
                                [1, 0, 0]])

        rotation = np.dot(rotation, permutation)

        return {'rotation': rotation, 'translation': translation}

    def get_parameters(self) -> dict:
        """
        Return the camera extrinsics as a dictionary.
        Expected keys: 'rotation' (3x3 numpy array) and 'translation' (3-element numpy array).
        """
        return self.extrinsics

    def get_rotation_matrix(self) -> np.ndarray:
        """
        Return the camera rotation matrix.
        """
        return self.extrinsics.get('rotation', np.eye(3))

    def get_translation_vector(self) -> np.ndarray:
        """
        Return the camera translation vector.
        """
        return self.extrinsics.get('translation', np.zeros((3,)))
