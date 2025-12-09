# Installation Guide

EdiHeadyTrack supports both **PyTorch** and **TensorFlow** for model inference. To keep the base installation clean and flexible, deep learning frameworks are optional dependencies.

## Prerequisites

- **Python:** Version 3.8 to 3.10 required.
- **GPU Usage:** NVIDIA GPU with latest drivers (for GPU acceleration).
- **Frameworks:** Install using the commands below for proper GPU support.

---

## 1. Core Package (CPU-Only)

Install the core library and essential scientific dependencies (NumPy, pandas, OpenCV, etc.):

```bash
pip install EdiHeadyTrack
```

---

## 2. Framework Installation Options (CPU/GPU)

It is highly recommended to use a virtual environment (`venv` or `conda`).

### Option A: TensorFlow / Keras Support

For TensorFlow/Keras-based models (including deepface):

```bash
pip install EdiHeadyTrack[tf-gpu]
```

- Installs TensorFlow 2.15, which will attempt to detect and use a compatible CUDA environment automatically.

---

### Option B: PyTorch Support

For PyTorch-based models (including ultralytics, 6DRepNet, VGGHeads):

**Step 1:** Install PyTorch for your CUDA version (visit [PyTorch official site](https://pytorch.org/get-started/locally/) for the latest command).

Example for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Step 2:** Install EdiHeadyTrack with PyTorch extras:

```bash
pip install EdiHeadyTrack[torch-gpu]
```

---

### Option C: Install Both Frameworks

If you need both TensorFlow and PyTorch:

1. Follow PyTorch GPU installation steps above.
2. Install combined extras:

```bash
pip install EdiHeadyTrack[all]
```

---

## Compatibility Check

Verify GPU access in your Python environment:

**PyTorch:**
```python
import torch
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
```

**TensorFlow:**
```python
import tensorflow as tf
print(f"TensorFlow GPU devices: {tf.config.list_physical_devices('GPU')}")
```

---

**Tip:** Always check official framework documentation for the latest installation instructions and CUDA compatibility.
