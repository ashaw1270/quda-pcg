#!/bin/bash -l
# Configure and build the quda-pcg fork with adaptive multigrid (AMG) plus the
# amg_pcg_ddag_baseline binary, which builds the MG hierarchy directly on the
# normal operator D^dag D (true Galerkin coarse operator) and uses it as the PCG
# preconditioner for D^dag D x = D^dag psi.
#
# Compilation does not need a GPU (only nvcc), so this can run on a login node.
# Running the resulting binary requires an A100 (sm_80) GPU node.
#
# Usage:
#   bash build_quda_mg.sh            # configure + build
#   JOBS=32 bash build_quda_mg.sh    # override parallel build jobs

set -euo pipefail

QUDA_SRC="${QUDA_SRC:-/lcrc/project/NeuPreCon/shawa/quda-pcg}"
BUILD_DIR="${BUILD_DIR:-${QUDA_SRC}/build-mg}"
GPU_ARCH="${GPU_ARCH:-sm_80}"          # swing = NVIDIA A100
CUDA_ARCH="${CUDA_ARCH:-80}"
JOBS="${JOBS:-32}"

echo "=== Loading modules ==="
module purge
module load gcc/11.4.0
module load cuda/12.6.0
module load cmake/3.30.2-ufv3dko
module list 2>&1 || true
echo "nvcc: $(command -v nvcc)"
nvcc --version | tail -2
echo "cmake: $(command -v cmake) ($(cmake --version | head -1))"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "=== Configuring (QUDA_MULTIGRID=ON, arch=${GPU_ARCH}) ==="
cmake "${QUDA_SRC}" \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DQUDA_GPU_ARCH="${GPU_ARCH}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
  -DQUDA_MULTIGRID=ON \
  -DQUDA_DIRAC_WILSON=ON \
  -DQUDA_DIRAC_CLOVER=ON \
  -DQUDA_DIRAC_TWISTED_MASS=OFF \
  -DQUDA_DIRAC_TWISTED_CLOVER=OFF \
  -DQUDA_DIRAC_CLOVER_HASENBUSCH=OFF \
  -DQUDA_DIRAC_NDEG_TWISTED_MASS=OFF \
  -DQUDA_DIRAC_NDEG_TWISTED_CLOVER=OFF \
  -DQUDA_DIRAC_STAGGERED=OFF \
  -DQUDA_DIRAC_DOMAIN_WALL=OFF \
  -DQUDA_DIRAC_LAPLACE=OFF \
  -DQUDA_QIO=OFF \
  -DQUDA_QMP=OFF \
  -DQUDA_MPI=OFF \
  -DQUDA_BUILD_ALL_TESTS=OFF \
  -DQUDA_BUILD_SHAREDLIB=ON

echo "=== Building amg_pcg_ddag_baseline, amg_materialize_minv, dirac_matvec_compare (jobs=${JOBS}) ==="
cmake --build . --target amg_pcg_ddag_baseline amg_materialize_minv dirac_matvec_compare -j "${JOBS}"

echo "=== Build complete ==="
echo "Binary: ${BUILD_DIR}/tests/amg_pcg_ddag_baseline"
echo "Binary: ${BUILD_DIR}/tests/amg_materialize_minv"
echo "Binary: ${BUILD_DIR}/tests/dirac_matvec_compare"
ls -la "${BUILD_DIR}/tests/amg_pcg_ddag_baseline" "${BUILD_DIR}/tests/amg_materialize_minv" "${BUILD_DIR}/tests/dirac_matvec_compare" 2>/dev/null || true
