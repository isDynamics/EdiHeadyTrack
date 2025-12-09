from .data import Data

class Infill:
    """
    A class for interpolating or upsampling Data so that it can match another sampling rate.
    """

    def __init__(self, target_rate: float = 100.0):
        """
        :param target_rate: Desired sampling rate (Hz)
        """
        self.target_rate = target_rate

    def upsample(self, data: Data) -> Data:
        """
        Interpolate or upsample the Data to match the target_rate.
        Dummy implementation: replace with real logic (e.g. resampling, interpolation).
        """
        # Example: create new time vector, interpolate euler angles and velocities
        return data
