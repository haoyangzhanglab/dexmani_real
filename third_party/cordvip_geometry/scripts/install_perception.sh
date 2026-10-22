#!/usr/bin/env bash
# Incremental setup for the existing real_robot environment; never opens devices.
set -euo pipefail

cordvip_root="$PWD"
if [[ "${1:-}" == "--workspace" ]]; then
    cordvip_root="$(cd -- "${2:?--workspace requires a path}" && pwd)"
    shift 2
fi
cordvip_library="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'USAGE'
Usage: conda activate real_robot
       bash third_party/cordvip_geometry/scripts/install_perception.sh [--workspace PATH]

Preserve the existing base environment, fetch missing official source checkouts
at recorded revisions, install incremental requirements, and build
nvdiffrast and FoundationPose mycpp. No model downloads or device access.
Existing source directories and local edits are reused without checkout or reset.
The workspace defaults to the current directory; --workspace must be the first option.
Logs and environment snapshots are saved under its reports/cordvip_install/.
USAGE
    exit 0
fi
if [[ $# -ne 0 ]]; then
    printf 'Unknown arguments; use --help.\n' >&2
    exit 2
fi

cd -- "$cordvip_root"
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    printf 'Activate the existing compatible conda environment before running this script.\n' >&2
    exit 1
fi
python - <<'PY'
import os
from pathlib import Path
import sys

if Path(sys.prefix).resolve() != Path(os.environ['CONDA_PREFIX']).resolve():
    raise SystemExit('python does not belong to the active conda environment')
PY

for cordvip_command in git cmake c++; do
    if ! command -v "$cordvip_command" >/dev/null; then
        printf 'Missing build tool: %s\n' "$cordvip_command" >&2
        exit 1
    fi
done
if [[ ! -x "$CONDA_PREFIX/bin/nvcc" || ! -d "$CONDA_PREFIX/targets/x86_64-linux/include" ]]; then
    printf 'The conda prefix must contain the CUDA compiler and target headers.\n' >&2
    exit 1
fi

mkdir -p reports/cordvip_install
cordvip_log_dir="$(mktemp -d "$cordvip_root/reports/cordvip_install/install_XXXXXX")"
exec > >(tee -a "$cordvip_log_dir/install.log") 2>&1
printf 'Installation log: %s\n' "$cordvip_log_dir"
python -m pip list --format=json > "$cordvip_log_dir/packages_before.json"
python - <<'PY' > "$cordvip_log_dir/base_constraints.txt"
from importlib.metadata import version
for package in ('torch', 'torchvision', 'numpy', 'transformers'):
    print(f'{package}=={version(package)}')
PY

checkout_source() {
    local cordvip_name="$1" cordvip_url="$2" cordvip_revision="$3"
    local cordvip_path="$cordvip_root/third_party/$cordvip_name"
    if [[ -L "$cordvip_path" ]]; then
        printf 'Refusing to modify an existing symlink: %s\n' "$cordvip_path" >&2
        return 1
    elif [[ ! -e "$cordvip_path" ]]; then
        git clone --filter=blob:none --no-checkout -- "$cordvip_url" "$cordvip_path"
        if ! git -C "$cordvip_path" cat-file -e "$cordvip_revision^{commit}"; then
            git -C "$cordvip_path" fetch --depth 1 origin "$cordvip_revision"
        fi
        git -C "$cordvip_path" checkout --detach "$cordvip_revision"
    elif [[ ! -d "$cordvip_path" ]]; then
        printf 'Source path is not a directory: %s\n' "$cordvip_path" >&2
        return 1
    fi
    if [[ -e "$cordvip_path/.git" ]]; then
        cordvip_revision="$(git -C "$cordvip_path" rev-parse HEAD)"
        git -C "$cordvip_path" status --short > "$cordvip_log_dir/${cordvip_name}_status.txt"
    else
        cordvip_revision="local_directory"
    fi
    printf '%s %s\n' "$cordvip_name" "$cordvip_revision" >> "$cordvip_log_dir/source_revisions.txt"
}

mkdir -p third_party
checkout_source Grounded-SAM-2 https://github.com/IDEA-Research/Grounded-SAM-2.git b7a9c29f196edff0eb54dbe14588d7ae5e3dde28
checkout_source Hunyuan3D-2.1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git 82920d643c0dc2f7bfd7255f45f62d386edfe60c
checkout_source FoundationPose https://github.com/NVlabs/FoundationPose.git a1b694b83e633c2cb6115b9063d940a687759392
checkout_source nvdiffrast https://github.com/NVlabs/nvdiffrast.git 253ac4fcea7de5f396371124af597e6cc957bfae

python -m pip install \
    --constraint "$cordvip_log_dir/base_constraints.txt" \
    --requirement "$cordvip_library/requirements-perception.txt"

# Build against the active conda CUDA toolkit; do not install upstream dependency
# lists that could replace the existing robot environment's Torch/CUDA versions.
export CUDA_HOME="$CONDA_PREFIX"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include"
export PATH="$CONDA_PREFIX/bin:$PATH"
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST=8.9
python -m pip install --no-build-isolation --no-deps \
    -e third_party/cordvip_geometry
python -m pip install --no-build-isolation --no-deps third_party/nvdiffrast

cordvip_pybind_dir="$(python -m pybind11 --cmakedir)"
cordvip_cmeel_prefix="$(python - <<'PY'
import sysconfig
from pathlib import Path
print(Path(sysconfig.get_path('purelib')) / 'cmeel.prefix')
PY
)"
cmake -S third_party/FoundationPose/mycpp -B third_party/FoundationPose/mycpp/build \
    -DPYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python" \
    -Dpybind11_DIR="$cordvip_pybind_dir" \
    -DCMAKE_PREFIX_PATH="$cordvip_cmeel_prefix"
cmake --build third_party/FoundationPose/mycpp/build -j4

python -m pip list --format=json > "$cordvip_log_dir/packages_after.json"
printf 'Dependency installation and builds completed. No models or devices were opened.\n'
printf 'Next, from the workspace root: python -m cordvip_geometry download\n'
printf 'FoundationPose weights require the separate official download described in docs/usage.md.\n'
