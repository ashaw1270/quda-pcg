#!/bin/bash -l
#PBS -A NeuPreCon
#PBS -l select=1:ngpus=1
#PBS -q backfill
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -N amg_materialize_minv
#
# Materialize dense AMG M^-1 matrices for SU3_eig_precond_compare.
#
# Submit or run interactively, e.g.:
#   L=4 RNG=37 INDICES=0 qsub run_amg_materialize.sh
#   L=4 RNG=37 INDICES=0 bash run_amg_materialize.sh
#
# Env knobs:
#   L, RNG, BETA, KAPPA, INDICES, GAUGE_DIR, OUT_DIR, QUDA_BUILD, MG_EXTRA

set -euo pipefail

FORK_DIR="${FORK_DIR:-/lcrc/project/NeuPreCon/shawa/quda-pcg}"
EIG_ROOT="${EIG_ROOT:-/lcrc/project/NeuPreCon/shawa/eigval-comp}"
JAX_DIR="${JAX_DIR:-${EIG_ROOT}/jax_model}"
QUDA_BUILD="${QUDA_BUILD:-${FORK_DIR}/build-mg}"
BIN="${BIN:-${QUDA_BUILD}/tests/amg_materialize_minv}"

L="${L:-4}"
RNG="${RNG:-37}"
BETA="${BETA:-5.5}"
KAPPA="${KAPPA:-0.276}"
INDICES="${INDICES:-0}"
MG_LEVELS="${MG_LEVELS:-2}"
PREC="${PREC:-double}"
PREC_SLOPPY="${PREC_SLOPPY:-single}"
PREC_PRECON="${PREC_PRECON:-single}"

GAUGE_DIR="${GAUGE_DIR:-${EIG_ROOT}/quda_gauges}"
OUT_DIR="${OUT_DIR:-${EIG_ROOT}/eig_compare_out/AMG-DdagD/minv}"
MG_EXTRA="${MG_EXTRA:---mg-smoother 0 cg --mg-smoother 1 cg \
--mg-coarse-solver 1 cg --mg-coarse-solver-maxiter 1 50 --mg-coarse-solver-tol 1 0.25 \
--mg-setup-inv 0 cg --mg-nvec 0 24 --mg-block-size 0 4 4 4 4 \
--mg-nu-pre 0 0 --mg-nu-post 0 4}"

# Optional symmetrization of the materialized AMG operator.
# Use matched --mg-nu-pre/--mg-nu-post for the best symmetry.
#
# --mg-symmetric-gs enables Gauss-Seidel smoothing, but by default it uses
# standard (non-linear) minimal-residual relaxation.  Add
# --mg-symmetric-gs-linear to make the smoother a fixed, linear, self-adjoint
# operator, which is what actually keeps the operator Hermitian.
#
# Symmetric Gauss-Seidel fine smoother only (linear/Hermitian):
#   MG_EXTRA="--mg-symmetric-gs 0 on --mg-symmetric-gs-linear 0 on \
#     --mg-nu-pre 0 2 --mg-nu-post 0 2"
#
# Fixed-degree Chebyshev coarse solve only:
#   MG_EXTRA="--mg-coarse-chebyshev 1 on --mg-coarse-chebyshev-degree 1 8 \
#     --mg-coarse-solver-cheby-basis-eig-min 1 1e-2 --mg-coarse-solver-cheby-basis-eig-max 1 10"
#
# Both together (fully symmetric V-cycle):
#   MG_EXTRA="--mg-symmetric-gs 0 on --mg-symmetric-gs-linear 0 on \
#     --mg-nu-pre 0 2 --mg-nu-post 0 2 \
#     --mg-coarse-chebyshev 1 on --mg-coarse-chebyshev-degree 1 8 \
#     --mg-coarse-solver-cheby-basis-eig-min 1 1e-2 --mg-coarse-solver-cheby-basis-eig-max 1 10"

echo "=== modules ==="
module purge
module load gcc/11.4.0 cuda/12.6.0
export LD_LIBRARY_PATH="${QUDA_BUILD}/lib:${LD_LIBRARY_PATH:-}"

echo "=== convert gauge (if needed) ==="
cd "${JAX_DIR}"
uv run python -m scripts.convert_gauge \
  --L "${L}" --rng "${RNG}" --beta "${BETA}" \
  --indices "${INDICES}" \
  --out-dir "${GAUGE_DIR}"

BETA_TAG="${BETA//./p}"
mapfile -t GAUGE_FILES < <(ls -1 "${GAUGE_DIR}"/L${L}_rng${RNG}_beta${BETA_TAG}_sample*_cfg*.qdp.bin 2>/dev/null | sort -t_ -k4.7n)
if [[ ${#GAUGE_FILES[@]} -eq 0 ]]; then
  mapfile -t GAUGE_FILES < <(ls -1 "${GAUGE_DIR}"/L${L}_rng${RNG}_beta${BETA}_sample*_cfg*.qdp.bin 2>/dev/null | sort -t_ -k4.7n)
fi
if [[ ${#GAUGE_FILES[@]} -eq 0 ]]; then
  echo "No gauge files found in ${GAUGE_DIR}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo "=== materialize Minv ==="
for gf in "${GAUGE_FILES[@]}"; do
  base="$(basename "${gf}")"
  if [[ "${base}" =~ cfg([0-9]+)\.qdp\.bin$ ]]; then
    cfg="${BASH_REMATCH[1]}"
  else
    echo "Could not parse config index from ${gf}" >&2
    exit 1
  fi
  bin_out="${OUT_DIR}/minv_cfg${cfg}.bin"
  npz_out="${OUT_DIR}/minv_cfg${cfg}.npz"
  echo "--- ${gf} -> ${npz_out} ---"
  set -x
  "${BIN}" \
    --dim "${L}" "${L}" "${L}" "${L}" \
    --dslash-type wilson \
    --kappa "${KAPPA}" \
    --fermion-t-boundary anti-periodic \
    --prec "${PREC}" \
    --prec-sloppy "${PREC_SLOPPY}" \
    --prec-precondition "${PREC_PRECON}" \
    --mg-levels "${MG_LEVELS}" \
    --verbosity verbose \
    ${MG_EXTRA} \
    --baseline-gauge-file "${gf}" \
    --baseline-config-index "${cfg}" \
    --materialize-out "${bin_out}"
  set +x
  uv run python -m scripts.bin_to_minv_npz "${bin_out}" "${npz_out}"
done

echo "=== done; Minv files in ${OUT_DIR} ==="
ls -la "${OUT_DIR}"
