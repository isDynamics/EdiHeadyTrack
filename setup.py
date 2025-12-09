from setuptools import setup, find_packages

# ----------------- Core deps: always installed -----------------
core_requires = [
    # Core scientific stack
    "numpy>=1.23,<3",
    "pandas>=1.5,<3",
    "scipy>=1.9,<2",
    "matplotlib>=3.6,<4",
    "seaborn>=0.12,<0.14",
    "tqdm>=4.66,<5",
    "PyYAML>=6.0,<7",
    "Pillow>=9.5,<12",
    "opencv-python>=4.8,<5",
    "trimesh>=4.0,<5",
]

# ----------------- Optional / heavy extras -----------------
extras = [
    # VGGHeads pipeline + batch-face (these pull in smplx, chumpy, etc.)
    "head_detector @ git+https://github.com/ThomasAston/VGGHeads.git@main",
    "batch_face @ git+https://github.com/elliottzheng/batch-face.git@master",
    "sixdrepnet @ git+https://github.com/ThomasAston/6DRepNet.git@master",
    "wandb>=0.16,<0.19",
    "thop>=0.1.1,<0.2",
    "ultralytics>=8.0.0,<9"
]

setup(
    name="EdiHeadyTrack",
    version="0.1.0",
    description="Benchmarking head pose estimation against sensor-based ground truth.",
    author="Thomas Aston",
    author_email="tjn.aston@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=core_requires,
    extras_require={
        "full": extras,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8,<3.12",
)
