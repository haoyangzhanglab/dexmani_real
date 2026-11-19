#!/usr/bin/env bash
# 在现有 real_robot 环境中安装依赖并编译扩展。
set -euo pipefail

cordvip_root="$PWD"
if [[ "${1:-}" == "--workspace" ]]; then
    cordvip_root="$(cd -- "${2:?--workspace requires a path}" && pwd)"
    shift 2
fi
cordvip_library="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cordvip_operation=tracking
if [[ "${1:-}" == "reconstruction" ]]; then
    cordvip_operation=reconstruction
    shift
fi

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'USAGE'
用法：conda activate real_robot
      bash third_party/cordvip_geometry/scripts/install_perception.sh [--workspace PATH] [reconstruction]

默认安装跟踪依赖，获取 Grounded-SAM-2、FoundationPose 和 nvdiffrast，
应用推理补丁并编译扩展。需要 conda CUDA 编译器及头文件、CMake、C++、
系统 Boost（system/program_options）和 OpenMP 开发文件。

reconstruction：在已完成跟踪安装的同一环境中获取 Hunyuan3D，
应用分层解码补丁并安装重建依赖。

源码按固定 revision 获取。已有目录须为 Git 工作树根目录，HEAD 匹配固定
revision 时复用本地修改。版本不符或补丁冲突时退出。
工作目录默认为当前目录，--workspace 必须放在首位。
安装不下载模型或访问设备，后续命令见完成提示。
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
cordvip_python="$CONDA_PREFIX/bin/python"

if ! command -v git >/dev/null; then
    printf 'Missing tool: git\n' >&2
    exit 1
fi
if [[ "$cordvip_operation" == tracking ]]; then
    for cordvip_command in cmake c++; do
        if ! command -v "$cordvip_command" >/dev/null; then
            printf 'Missing build tool: %s\n' "$cordvip_command" >&2
            exit 1
        fi
    done
    if [[ ! -x "$CONDA_PREFIX/bin/nvcc" || ! -d "$CONDA_PREFIX/targets/x86_64-linux/include" ]]; then
        printf 'The conda prefix must contain the CUDA compiler and target headers.\n' >&2
        exit 1
    fi
fi

cordvip_constraints="$(mktemp)"
trap 'rm -f "$cordvip_constraints"' EXIT
"$cordvip_python" - <<'PY' > "$cordvip_constraints"
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
    else
        local cordvip_worktree cordvip_head
        cordvip_path="$(cd -- "$cordvip_path" && pwd -P)"
        if ! cordvip_worktree="$(git -C "$cordvip_path" rev-parse --show-toplevel)" ||
                [[ "$cordvip_worktree" != "$cordvip_path" ]]; then
            printf 'Expected a Git checkout at %s\n' "$cordvip_path" >&2
            return 1
        fi
        cordvip_head="$(git -C "$cordvip_path" rev-parse --verify 'HEAD^{commit}')"
        if [[ "$cordvip_head" != "$cordvip_revision" ]]; then
            printf 'Source revision mismatch at %s (expected %s, found %s).\n' \
                "$cordvip_path" "$cordvip_revision" "$cordvip_head" >&2
            return 1
        fi
    fi
}

apply_source_patch() {
    local cordvip_source="$cordvip_root/third_party/$1"
    local cordvip_patch="$cordvip_library/scripts/$2"
    if ! git -C "$cordvip_source" apply --reverse --check "$cordvip_patch" 2>/dev/null; then
        git -C "$cordvip_source" apply "$cordvip_patch"
    fi
}

mkdir -p third_party
if [[ "$cordvip_operation" == reconstruction ]]; then
    checkout_source Hunyuan3D-2.1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git 82920d643c0dc2f7bfd7255f45f62d386edfe60c
    apply_source_patch Hunyuan3D-2.1 hunyuan_hierarchical_decode.patch
    "$cordvip_python" -m pip install \
        --constraint "$cordvip_constraints" \
        diffusers==0.30.3 \
        pymeshlab==2023.12.post3 \
        scikit-image==0.24.0
    printf '重建依赖安装完成。下载模型：python -m cordvip_geometry download --models shape\n'
else
    checkout_source Grounded-SAM-2 https://github.com/IDEA-Research/Grounded-SAM-2.git b7a9c29f196edff0eb54dbe14588d7ae5e3dde28
    checkout_source FoundationPose https://github.com/NVlabs/FoundationPose.git a1b694b83e633c2cb6115b9063d940a687759392
    checkout_source nvdiffrast https://github.com/NVlabs/nvdiffrast.git 253ac4fcea7de5f396371124af597e6cc957bfae

    apply_source_patch Grounded-SAM-2 sam2_inference.patch
    apply_source_patch FoundationPose foundationpose_inference.patch

    "$cordvip_python" -m pip install \
        --constraint "$cordvip_constraints" \
        hydra-core==1.3.5 \
        iopath==0.1.10 \
        ninja==1.13.2 \
        kornia==0.7.3 \
        warp-lang==1.5.1 \
        ruamel.yaml==0.18.6 \
        transformations==2024.5.24 \
        cmeel-eigen==3.4.1 \
        pybind11==2.13.1

    # 使用现有 conda CUDA 编译，保留基础 Torch/CUDA 版本。
    export CUDA_HOME="$CONDA_PREFIX"
    export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
    export PATH="$CONDA_PREFIX/bin:$PATH"
    export MAX_JOBS="${MAX_JOBS:-4}"
    "$cordvip_python" -m pip install --no-build-isolation \
        --constraint "$cordvip_constraints" -e "${cordvip_library}[cloud]"
    "$cordvip_python" -m pip install --no-build-isolation --no-deps third_party/nvdiffrast

    cordvip_pybind_dir="$("$cordvip_python" -m pybind11 --cmakedir)"
    cordvip_cmeel_prefix="$("$cordvip_python" - <<'PY'
import sysconfig
from pathlib import Path
print(Path(sysconfig.get_path('purelib')) / 'cmeel.prefix')
PY
    )"
    cmake -S third_party/FoundationPose/mycpp -B third_party/FoundationPose/mycpp/build \
        -DPYTHON_EXECUTABLE="$cordvip_python" \
        -Dpybind11_DIR="$cordvip_pybind_dir" \
        -DCMAKE_PREFIX_PATH="$cordvip_cmeel_prefix"
    cmake --build third_party/FoundationPose/mycpp/build -j"$MAX_JOBS"
    printf '跟踪依赖编译完成。下载模型：python -m cordvip_geometry download\n'
    printf 'FoundationPose 权重另放于 third_party/FoundationPose/weights/，见后端 README。\n'
fi
