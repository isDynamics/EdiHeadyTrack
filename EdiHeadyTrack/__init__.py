# Lightweight core that should work in any env
from .camera import Camera
from .data import Data
from .filter import SO3Filter
from .plot import Plot
from .sensor import SensorDataLoader
from .video import Video
from .analyse import Analyser
from .alignment import align_to_gt
from .benchmark import Benchmark
from .infill import Infill

__all__ = [
    "Camera",
    "Data",
    "SO3Filter",
    "Plot",
    "SensorDataLoader",
    "Video",
    "Analyser",
    "align_to_gt",
    "Benchmark",
    "Infill",
]

# Optional / heavy pieces that may require extras (VGGHeads, sixdrepnet, etc.)
# They will only be available if the corresponding extras are installed.

try:
    from .head_pose import HeadPoseModel
except ImportError:
    HeadPoseModel = None
else:
    __all__.append("HeadPoseModel")

try:
    from .head_detection import HeadDetector
except ImportError:
    HeadDetector = None
else:
    __all__.append("HeadDetector")

# If you actually have a ReferenceAligner class in some module, import it here.
# If not, it's safer to just remove it from __all__ entirely.
# Example (uncomment / adjust if it exists):
# try:
#     from .alignment import ReferenceAligner
# except ImportError:
#     ReferenceAligner = None
# else:
#     __all__.append("ReferenceAligner")
