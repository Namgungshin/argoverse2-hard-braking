#!/usr/bin/env bash
# Mac 터미널에서 이 폴더로 이동한 뒤 실행: bash setup_and_download.sh
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# s5cmd 설치 (Homebrew)
brew list s5cmd >/dev/null 2>&1 || brew install s5cmd

# val 먼저 받기 (용량이 작아 파이프라인 검증용으로 적합)
mkdir -p data/val
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/motion-forecasting/val/*" data/val/
echo "완료: $(ls data/val | wc -l)개 시나리오 폴더"
