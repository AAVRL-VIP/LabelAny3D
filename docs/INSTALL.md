# Installation

This guide installs LabelAny3D with two conda environments:

- `labelany3d`: the main pipeline environment
- `sam`: the separate SAM3 environment

Start by cloning the repository.

## 1. Clone the Repository

Clone LabelAny3D and enter the repository.

```bash
git clone https://github.com/AAVRL-VIP/LabelAny3D.git
cd LabelAny3D
```

## 2. Set the External Directory

This variable points to the repository's `external` directory. The later commands use it to install MoGe, DepthPro, checkpoints, Amodal3R, and TRELLIS.

```bash
export EXT_DIR=$(pwd)/external
```

## 3. Create the Main Pipeline Environment

Create the main Python 3.10 environment for LabelAny3D.

```bash
conda create -n labelany3d python=3.10 -y
conda activate labelany3d
```

Install PyTorch and Torchvision for CUDA 12.1 wheels. This environment is used for the main LabelAny3D pipeline.

```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
```

## 4. Install Base LabelAny3D Requirements

Install Diffusers and the repository requirements.

```bash
pip install diffusers==0.30.0
pip install -r requirements.txt
```

Install PyTorch3D, Detectron2, and pycocotools.

```bash
pip install git+https://github.com/facebookresearch/pytorch3d.git@055ab3a --no-build-isolation
pip install git+https://github.com/yaojin17/detectron2.git --no-build-isolation
pip install pycocotools==2.0.11 --no-build-isolation
```

## 5. Install MoGe

MoGe is used by the geometry/depth-related parts of the pipeline.

```bash
cd $EXT_DIR/MoGe
pip install -r requirements.txt
```

## 6. Install DepthPro

DepthPro is installed from the local external source tree.

```bash
cd $EXT_DIR/ml-depth-pro
pip install -e .
```

## 7. Download Checkpoints

Download the external checkpoint files used by the pipeline.

```bash
cd $EXT_DIR/checkpoints
bash download.sh
```

## 8. Download Amodal3R

Amodal3R is used as the 3D reconstruction backend.

```bash
cd $EXT_DIR

git clone https://github.com/Sm0kyWu/Amodal3R.git Amodal3R
cd Amodal3R
```
Set the C and C++ compilers before building the remaining CUDA extensions.

```bash
export CC=$(which gcc)
export CXX=$(which g++)
```

## 9. Install TRELLIS CUDA Extensions

TRELLIS provides CUDA extensions required by the 3D reconstruction stack.

```bash
cd $EXT_DIR/TRELLIS
. ./setup.sh --basic --xformers --diffoctreerast --spconv --mipgaussian --nvdiffrast
```

## 10. Install Amodal3R Runtime Dependencies

Install xFormers for the PyTorch 2.2.2 CUDA 12.1 environment.

```bash
python -m pip install "xformers==0.0.25.post1" --index-url https://download.pytorch.org/whl/cu121 --no-deps
```

Install nvdiffrast.

```bash
python -m pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
```

Install the Gaussian rasterization dependency used by Amodal3R.

```bash
python -m pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization --no-build-isolation
```


## 11. Install flash-attn

Download and install the flash-attn wheel matching Python 3.10, PyTorch 2.2, and CUDA 12.

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install "setuptools<70" wheel packaging

wget -O /tmp/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install /tmp/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

rm -f /tmp/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## 12. Create the SAM3 Environment

Create the separate SAM3 conda environment from `envs/sam_conda_explicit.txt`.

```bash
cd path/to/LabelAny3D
conda create -n sam --file envs/sam_conda_explicit.txt -y
conda activate sam
```

Install and upgrade pip tooling inside the SAM3 environment.

```bash
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
```

Install the SAM3 Python requirements.

```bash
python -m pip install -r envs/sam_requirements.txt
```

## 13. Log in to Hugging Face

SAM3 requires Hugging Face authentication for gated model files.

```bash
conda activate sam
hf auth login
```

## 14. Install Web Demo Requirements

Install the web demo requirements.

```bash
pip install -r web_demo/requirements.txt

