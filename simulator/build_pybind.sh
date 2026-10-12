#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source training/.venv/bin/activate
cd simulator

INCLUDES=$(python3 -m pybind11 --includes)
SUFFIX=$(python3-config --extension-suffix)

g++ -O3 -Wall -Wextra -shared -std=c++17 -fPIC $INCLUDES pybind_module.cpp -o "tronsim${SUFFIX}"
echo "Built tronsim${SUFFIX}"
ls -la tronsim*.so
