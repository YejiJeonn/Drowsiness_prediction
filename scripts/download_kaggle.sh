#!/usr/bin/env bash
set -e

mkdir -p data/raw/mrl_eye data/raw/yawdd data/raw/eye_open_close

kaggle datasets download -d akashshingha850/mrl-eye-dataset -p data/raw/mrl_eye --unzip
kaggle datasets download -d enider/yawdd-dataset -p data/raw/yawdd --unzip
kaggle datasets download -d shuvokumarbasak4004/eye-open-close -p data/raw/eye_open_close --unzip

echo "Kaggle datasets downloaded into data/raw/"
