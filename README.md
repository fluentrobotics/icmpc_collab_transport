# Implicit Communication in Human-Robot Collaborative Transport

[![arxiv](https://img.shields.io/badge/Preprint-gray?logo=arxiv&labelColor=B31B1B)
](https://arxiv.org/abs/2502.03346)
[![dataset](<https://img.shields.io/badge/Dataset-gray?logo=dropbox&labelColor=0061FF>)](https://www.dropbox.com/scl/fo/ruzihfyim9n5wx3kcluc6/APV1wxgUifVPQnd12RSqjbo?rlkey=u1d4pdei9jruil1bl4j0fxkmf&st=1e7mman0&dl=0)
[![video](https://img.shields.io/badge/Supplemental_Video-gray?logo=youtube&labelColor=ff0033)](https://youtu.be/0NTDrobSifg)

## Requirements

### Core

* Python 3.10
* Poetry package manager
* ~10 GB disk space for Python packages

### Extra Dependencies for Deployment and Data Analysis

* ROS 2 Humble (Ubuntu 22.04)
* GNU tar, zstd, and ~20 GB disk space for dataset

## Setup

### Core

Install Python package dependencies to a virtual environment:

```shell
poetry install
```

Activate the virtual environment:

```shell
source .venv/bin/activate
```

### Dataset

Download `rosbags.tar.zst` from the Dropbox Dataset link in this file's header into [`data/fluentrobotics`](data/fluentrobotics).

Decompress the tar file:

```shell
cd data/fluentrobotics
tar xf rosbags.tar.zst
```

## Repository Layout

```text
.
├── data/
│   └── fluentrobotics
│       (Data files from our user study)
│
├── deployment_scripts/
│   (Convenience scripts used during our user study)
│
├── src/
│   ├── eleyng/
│   │   └── table-carrying-ai/
│   │       (3rd party code required for the VRNN baseline, see README.md in this folder)
│   │
│   └── fluentrobotics/
│       └── icmpc_collab_transport/
│           ├── core/
│           │   (Implementation of the core inference algorithm)
│           │
│           ├── deployment/
│           │   (ROS 2 nodes to run algorithms on a Hello Robot Stretch 2)
│           │
│           └── evaluation/
│               (Modules to analyze and visualize data from our user study)
│
├── pyproject.toml
│   (Specification file for Python package dependencies)
│
└── README.md
    (📍 You are here)
```

## Citation

```bibtex
@article{yang2025implicit,
  title={Implicit Communication in Human-Robot Collaborative Transport},
  author={Yang, Elvin and Mavrogiannis, Christoforos},
  journal={arXiv preprint arXiv:2502.03346},
  year={2025}
}
```
