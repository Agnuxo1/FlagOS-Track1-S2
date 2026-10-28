#!/usr/bin/env bash
# Runs tests + benchmark on the GPU server; outputs land in results/.
set -euo pipefail
cd "$(dirname "$0")/.."
export TRITON_CACHE_DIR="$PWD/.triton_cache"
python -c "import torch,triton;print('torch',torch.__version__,'triton',triton.__version__,torch.cuda.get_device_name(0))"
python -m pytest -q tests/test_kernels.py 2>&1 | tee results/pytest_$(date -u +%Y%m%dT%H%M%SZ).txt
python bench/bench.py --runs 3 --rounds 5 "$@" 2>&1 | tee results/bench_log_$(date -u +%Y%m%dT%H%M%SZ).txt
