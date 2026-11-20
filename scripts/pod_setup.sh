#!/bin/bash
# One-shot setup on a fresh GPU box (tested on RunPod image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404).
# Installs LucidFace and fetches third-party models. Each model keeps ITS OWN licence (see NOTICE.md):
#   CodeFormer  - S-Lab License 1.0, NON-COMMERCIAL
#   Restormer   - MIT (only needed for --body)
#   GFPGAN      - Apache-2.0 code, weights trained on FFHQ (only needed for --backend gfpgan)
#   facenet     - MIT code, VGGFace2-trained weights (only needed for --reference)
# usage: bash scripts/pod_setup.sh [/path/to/models]
set -e
M=${1:-/root/models}
export PIP_BREAK_SYSTEM_PACKAGES=1
mkdir -p "$M"
pip install -q -e ".[bench,dev]"
pip install -q --no-deps facenet-pytorch
pip install -q gdown

if [ ! -d "$M/CodeFormer" ]; then
  git clone --depth 1 https://github.com/sczhou/CodeFormer "$M/CodeFormer"
  R=https://github.com/sczhou/CodeFormer/releases/download/v0.1.0
  wget -q -O "$M/CodeFormer/weights/CodeFormer/codeformer.pth" $R/codeformer.pth
fi
if [ ! -d "$M/Restormer" ]; then
  git clone --depth 1 https://github.com/swz30/Restormer "$M/Restormer"
  # weights live in a Google Drive folder linked from Defocus_Deblurring/pretrained_models/README.md
  (cd "$M/Restormer/Defocus_Deblurring/pretrained_models" && \
   gdown --folder https://drive.google.com/drive/folders/1bRBG8DG_72AGA6-eRePvChlT5ZO4cwJ4 -O . || \
   echo "Restormer weights: download single_image_defocus_deblurring.pth by hand into this folder")
fi
if [ "${WITH_GFPGAN:-0}" = 1 ]; then
  pip install -q gfpgan
  F=$(python -c "import basicsr,os;print(os.path.dirname(basicsr.__file__))")/data/degradations.py
  sed -i "s/torchvision.transforms.functional_tensor/torchvision.transforms.functional/" "$F" || true
  wget -q -nc -P "$M" https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
fi
cat <<EOF
Setup done. Add to your shell:
  export LUCIDFACE_CODEFORMER_DIR=$M/CodeFormer
  export LUCIDFACE_RESTORMER_DIR=$M/Restormer
  export LUCIDFACE_GFPGAN_WEIGHTS=$M/GFPGANv1.4.pth
EOF
