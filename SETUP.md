# Installation

Install [OmniSeg3D](https://github.com/OceanYing/OmniSeg3D-GS).

The **Gaussian Splatting** version is easier to install; however, the method also works with the original **NeRF-based** version.
I tested it with **Python 3.11** and **PyTorch 2**, so you may need to adjust dependencies accordingly.

If you plan to use **PyTorch 2.x**, apply [this patch](./code/pytorch_2_0.diff).
You may also find [this conda environment file](./code/omniseg3dgs_ovs.yaml) useful.

---

# Running a Demo

To run the model on the first scene from the Replica dataset, download:

1. [Reconstructed scene](..)
2. [LSeg-processed renders](..)
3. [OmniSeg3D renders](..) (or generate them locally by running the renderer)

Then:

1. Apply [this patch](./code/ov_search_demo.diff).
2. Add [this file](./code/open_vocabulary_search.py) to enable **open-vocabulary search** in the existing OmniSeg3D GUI.
3. Update the scene path in `render_omni_gui.py` to point to your downloaded files and directories.

> **Note:** You can speed up clustering by replacing the CPU-based **HDBSCAN** implementation with the GPU-accelerated version from [RAPIDS cuML](https://developer.nvidia.com/blog/faster-hdbscan-soft-clustering-with-rapids-cuml/).
> However, RAPIDS is **Linux-only**. On Windows, you can use WSL or Docker.
> The provided code uses CPU-based clustering for better portability.

---

# Training

To process a custom scene:

1. Convert your input into an **OmniSeg3D-compatible format** (e.g., using **COLMAP**).
   Follow the [data preparation instructions](https://github.com/OceanYing/OmniSeg3D-GS?tab=readme-ov-file#data-preparation).
2. Train and render following the [OmniSeg3D instructions](https://github.com/OceanYing/OmniSeg3D-GS?tab=readme-ov-file#training).
3. Process the data with **LSeg**.
   For this, I recommend using [this notebook]() on Google Colab.

---

# Google Colab

If you want to reproduce the workflow on **Google Colab**, the following notebooks may help:

1. [OmniSeg3D-GS Training & Rendering](..)
2. [LSeg Processing](..)
