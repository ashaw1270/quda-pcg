#!/bin/bash -l
#PBS -A NeuPreCon
#PBS -l select=1:ngpus=4
#PBS -q backfill
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -N amg_pcg_ddag_baseline
#
# Run the Galerkin-AMG-preconditioned PCG baseline (D^dag D x = D^dag psi) on one
# GPU and append per-config metrics to a CSV with the MatrixPreNet
# test_metrics schema. The MG hierarchy is built directly on the normal operator
# D^dag D (true Galerkin coarse operator, two levels), so one V-cycle ~ (D^dag
# D)^{-1} is the PCG preconditioner.
#
# Submit, e.g.:
#   L=8  RNG=31 qsub run_amg_ddag_baseline.sh
#   L=16 RNG=37 qsub run_amg_ddag_baseline.sh
# or run inside an interactive GPU job:
#   L=8  RNG=31 bash run_amg_ddag_baseline.sh
#
# Env knobs (with defaults):
#   L, RNG, BETA, KAPPA, MG_LEVELS(=2), PREC, PREC_SLOPPY, PREC_PRECON,
#   TOL, NITER, NRHS(=20), NCONFIG(=20), MODEL_NAME, METRICS_ROOT, PER_MODEL_CSV,
#   AGGREGATE_CSV, GAUGE_DIR, SEED, MG_EXTRA
#   Set NCONFIG=0 to use every available converted gauge file.

set -euo pipefail

FORK_DIR="${FORK_DIR:-/lcrc/project/NeuPreCon/shawa/quda-pcg}"
BASE_DIR="${BASE_DIR:-/lcrc/project/NeuPreCon/shawa/MatrixPreNet/quda_baseline}"
QUDA_BUILD="${QUDA_BUILD:-${FORK_DIR}/build-mg}"
BIN="${BIN:-${QUDA_BUILD}/tests/amg_pcg_ddag_baseline}"

L="${L:-8}"
RNG="${RNG:-31}"
BETA="${BETA:-5.5}"
KAPPA="${KAPPA:-0.276}"
MG_LEVELS="${MG_LEVELS:-2}"               # Galerkin normal-op MG is two-level only
PREC="${PREC:-double}"
PREC_SLOPPY="${PREC_SLOPPY:-single}"      # QUDA does not compile double-precision MG by default
PREC_PRECON="${PREC_PRECON:-single}"
TOL="${TOL:-1e-8}"
NITER="${NITER:-300}"
SEED="${SEED:-1234}"
NRHS="${NRHS:-20}"                         # random RHS solved per gauge config
NCONFIG="${NCONFIG:-20}"                   # gauge configs to run; 0 = all available

# D^dag D is Hermitian positive-definite, so use CG for the smoother, the coarse
# solve and the null-space setup. Tunable via MG_EXTRA.
MG_EXTRA="${MG_EXTRA:---mg-smoother 0 cg --mg-smoother 1 cg \
--mg-coarse-solver 1 cg --mg-coarse-solver-maxiter 1 50 --mg-coarse-solver-tol 1 0.25 \
--mg-setup-inv 0 cg --mg-nvec 0 24 --mg-block-size 0 4 4 4 4 \
--mg-nu-pre 0 0 --mg-nu-post 0 4}"

# Optional symmetrization of the AMG preconditioner (set MG_EXTRA to enable).
# Use matched --mg-nu-pre/--mg-nu-post for the best symmetry.
#
# --mg-symmetric-gs enables Gauss-Seidel smoothing, but by default it uses
# standard (non-linear) minimal-residual relaxation.  Add
# --mg-symmetric-gs-linear to make the smoother a fixed, linear, self-adjoint
# operator, which is what actually keeps the preconditioner Hermitian.
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

MODEL_NAME="${MODEL_NAME:-AMG}"
GAUGE_DIR="${GAUGE_DIR:-${BASE_DIR}/gauges}"
METRICS_ROOT="${METRICS_ROOT:-/lcrc/project/NeuPreCon/shawa/ExperimentLogs}"
PER_MODEL_CSV="${PER_MODEL_CSV:-${METRICS_ROOT}/per_model_metrics/${MODEL_NAME}.csv}"
AGGREGATE_CSV="${AGGREGATE_CSV:-${METRICS_ROOT}/aggregate_test_metrics.csv}"
mkdir -p "$(dirname "${PER_MODEL_CSV}")" "$(dirname "${AGGREGATE_CSV}")"

echo "=== modules ==="
module purge
module load gcc/11.4.0 cuda/12.6.0
export LD_LIBRARY_PATH="${QUDA_BUILD}/lib:${LD_LIBRARY_PATH:-}"

# Collect gauge files for this (L, RNG, BETA) in sample order.
mapfile -t GAUGE_FILES < <(ls -1 "${GAUGE_DIR}"/L${L}_rng${RNG}_beta${BETA}_sample*.qdp.bin 2>/dev/null | sort -t_ -k4.7n)
if [[ ${#GAUGE_FILES[@]} -eq 0 ]]; then
  echo "No gauge files found in ${GAUGE_DIR} for L=${L} rng=${RNG} beta=${BETA}" >&2
  echo "Run convert_gauge.py first." >&2
  exit 1
fi
if [[ "${NCONFIG}" -gt 0 ]]; then
  GAUGE_FILES=("${GAUGE_FILES[@]:0:${NCONFIG}}")
fi
echo "Found ${#GAUGE_FILES[@]} gauge config(s):"
printf '  %s\n' "${GAUGE_FILES[@]}"

GAUGE_ARGS=()
for f in "${GAUGE_FILES[@]}"; do GAUGE_ARGS+=(--baseline-gauge-file "$f"); done

echo "=== running amg_pcg_ddag_baseline ==="
set -x
"${BIN}" \
  --dim "${L}" "${L}" "${L}" "${L}" \
  --dslash-type wilson \
  --kappa "${KAPPA}" \
  --fermion-t-boundary anti-periodic \
  --prec "${PREC}" \
  --prec-sloppy "${PREC_SLOPPY}" \
  --prec-precondition "${PREC_PRECON}" \
  --tol "${TOL}" \
  --niter "${NITER}" \
  --mg-levels "${MG_LEVELS}" \
  --verbosity verbose \
  ${MG_EXTRA} \
  "${GAUGE_ARGS[@]}" \
  --baseline-per-model-csv "${PER_MODEL_CSV}" \
  --baseline-aggregate-csv "${AGGREGATE_CSV}" \
  --baseline-model-name "${MODEL_NAME}" \
  --baseline-beta "${BETA}" \
  --baseline-rng "${RNG}" \
  --baseline-sample-base 0 \
  --baseline-num-rhs "${NRHS}" \
  --baseline-seed "${SEED}"
set +x

echo "=== done; per-model metrics written to ${PER_MODEL_CSV} ==="
cat "${PER_MODEL_CSV}" || true
echo "=== aggregate row updated in ${AGGREGATE_CSV} ==="
cat "${AGGREGATE_CSV}" || true
