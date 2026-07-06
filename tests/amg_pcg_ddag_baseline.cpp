// Galerkin-AMG-preconditioned PCG baseline for A x = b with A = D^dag D,
// b = D^dag psi, using a true two-level multigrid built on the normal operator.
//
// Unlike amg_pcg_baseline (which builds the hierarchy on the direct operator D
// and adapts it to D^dag D), this driver enables QUDA_MG_NORMAL_GALERKIN so the
// MG hierarchy is built directly on A = D^dag D:
//   * fine inter-grid residual, smoother and null-space operators use DiracMdagM
//   * the coarse operator is the matrix-free Galerkin product P^dag (D^dag D) P
//     (see DiracGalerkinNormal / MG::createCoarseDirac).
// One V-cycle therefore approximates (D^dag D)^{-1} directly, so the PCG
// preconditioner is simply z = mg(r) (no gamma5 trick, no inner solve).
//
// Two levels only: the matrix-free Galerkin coarse operator cannot be
// re-coarsened by QUDA's link kernels.
//
// A = D^dag D is never formed: it is applied matrix-free as two Wilson-Dirac
// matvecs via Dirac::MdagM, and b = D^dag psi via Dirac::Mdag.  The PCG loop
// mirrors MatrixPreNet/src/utils/solver.py (zero initial guess, z = M(r),
// relative residual ||r|| / ||b||).
//
// Timing is setup-inclusive and fair vs the neural preconditioners: per gauge
// config it records the AMG hierarchy build cost (setup_cost_s) and the
// one-time QUDA autotuning cost (warmup_s) separately, then solves
// --baseline-num-rhs random right-hand sides reusing that hierarchy, timing
// each solve. A per-model CSV (--baseline-per-model-csv) holds one setup row
// plus one row per RHS; a shared aggregate CSV (--baseline-aggregate-csv) holds
// one averaged row per run mirroring the neural side's schema.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <chrono>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

#include <quda.h>
#include <color_spinor_field.h>
#include <dirac_quda.h>
#include <multigrid.h>
#include <blas_quda.h>

#include "misc.h"
#include "host_utils.h"
#include "command_line_params.h"

using quda::ColorSpinorField;
using quda::ColorSpinorParam;

// QUDA test-side global parameter structs.
QudaGaugeParam gauge_param;
QudaInvertParam inv_param;
QudaMultigridParam mg_param;
QudaInvertParam mg_inv_param;
QudaEigParam mg_eig_param[QUDA_MAX_MG_LEVEL];

// Host gauge storage (QDP order: gauge[mu] is V * gauge_site_size reals).
std::vector<char> gauge_;
std::array<void *, 4> gauge;

// ---- baseline-specific command line options ------------------------------
std::vector<std::string> baseline_gauge_files;
std::string baseline_per_model_csv = "per_model_metrics.csv";
std::string baseline_aggregate_csv = "aggregate_test_metrics.csv";
std::string baseline_model_name = "AMG";
double baseline_beta = 5.5;
int baseline_rng = 0;
int baseline_sample_base = 0;
int baseline_num_rhs = 20;
unsigned long baseline_seed = 1234UL;

static void load_gauge_binary(const std::string &path)
{
  FILE *fp = fopen(path.c_str(), "rb");
  if (!fp) errorQuda("Could not open gauge file %s", path.c_str());
  const size_t per_dir = (size_t)V * gauge_site_size;
  for (int mu = 0; mu < 4; mu++) {
    double *dst = reinterpret_cast<double *>(gauge[mu]);
    size_t got = fread(dst, sizeof(double), per_dir, fp);
    if (got != per_dir)
      errorQuda("Short read on %s (mu=%d): got %zu of %zu doubles", path.c_str(), mu, got, per_dir);
  }
  long pos = ftell(fp);
  fseek(fp, 0, SEEK_END);
  long end = ftell(fp);
  fclose(fp);
  if (pos != end) errorQuda("Gauge file %s has %ld trailing bytes; layout mismatch", path.c_str(), end - pos);
}

static std::string iso_timestamp()
{
  std::time_t t = std::time(nullptr);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
  return std::string(buf);
}

static double round_4sig(double x)
{
  if (x == 0.0 || !std::isfinite(x)) return x;
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.4g", x);
  return std::strtod(buf, nullptr);
}

static std::string format_metric_str(double x)
{
  const double r4dp = std::round(x * 10000.0) / 10000.0;
  const double r4sig = round_4sig(x);
  char buf[64];
  if (std::abs(x - r4dp) <= std::abs(x - r4sig))
    std::snprintf(buf, sizeof(buf), "%.4f", r4dp);
  else
    std::snprintf(buf, sizeof(buf), "%.4g", r4sig);
  return std::string(buf);
}

static const char *PER_MODEL_HEADER =
  "timestamp,model,L,beta,kappa,rng,sample,phase,rhs_idx,elapsed_s,warmup_s,iters,final_resid,converged\n";

// Truncate the per-model CSV and write its header so each run regenerates it.
static void init_per_model_csv(const std::string &csv_path)
{
  FILE *f = fopen(csv_path.c_str(), "w");
  if (!f) errorQuda("Could not open per-model CSV %s for write", csv_path.c_str());
  fputs(PER_MODEL_HEADER, f);
  fclose(f);
}

// One setup row per gauge: elapsed_s = setup (AMG hierarchy build) cost,
// warmup_s = one-time QUDA autotuning; the rhs_idx/iters/resid/converged
// columns are left blank.
static void append_setup_row(const std::string &csv_path, int L, int sample, double setup_cost_s, double warmup_s)
{
  const std::string setup_s = format_metric_str(setup_cost_s);
  const std::string warm_s = format_metric_str(warmup_s);
  FILE *f = fopen(csv_path.c_str(), "a");
  if (!f) errorQuda("Could not open per-model CSV %s for append", csv_path.c_str());
  fprintf(f, "%s,%s,%d,%g,%g,%d,%d,setup,,%s,%s,,,\n", iso_timestamp().c_str(), baseline_model_name.c_str(), L,
          baseline_beta, kappa, baseline_rng, sample, setup_s.c_str(), warm_s.c_str());
  fclose(f);
}

// One row per solved right-hand side: elapsed_s = solve time; the warmup_s
// column is left blank.
static void append_rhs_row(const std::string &csv_path, int L, int sample, int rhs_idx, double elapsed_s, int iters,
                           double final_resid, bool converged)
{
  const std::string elapsed = format_metric_str(elapsed_s);
  const std::string resid = format_metric_str(final_resid);
  FILE *f = fopen(csv_path.c_str(), "a");
  if (!f) errorQuda("Could not open per-model CSV %s for append", csv_path.c_str());
  fprintf(f, "%s,%s,%d,%g,%g,%d,%d,rhs,%d,%s,,%d,%s,%s\n", iso_timestamp().c_str(), baseline_model_name.c_str(), L,
          baseline_beta, kappa, baseline_rng, sample, rhs_idx, elapsed.c_str(), iters, resid.c_str(),
          converged ? "True" : "False");
  fclose(f);
}

// Append one averaged row to the shared aggregate CSV.
static void append_aggregate_csv(const std::string &csv_path, int L, double avg_iters, double setup_cost_s,
                                 double rhs_cost_s, double warmup_s, double total_test_s, double avg_final_resid,
                                 double frac_converged)
{
  static const char *header =
    "timestamp,model,L,beta,kappa,rng,iters,warmup_s,setup_cost_s,total_init_s,RHS_cost_s,TOTAL_s,final_resid,converged %\n";

  bool write_header = true;
  {
    std::ifstream in(csv_path);
    if (in && in.peek() != std::ifstream::traits_type::eof()) write_header = false;
  }

  const std::string warm_s = format_metric_str(warmup_s);
  const std::string setup_s = format_metric_str(setup_cost_s);
  const std::string total_s = format_metric_str(setup_cost_s + warmup_s);
  const std::string rhs_s = format_metric_str(rhs_cost_s);
  const std::string total_test_str = format_metric_str(total_test_s);
  const std::string resid_s = format_metric_str(avg_final_resid);
  const std::string conv_s = format_metric_str(frac_converged * 100.0);

  FILE *f = fopen(csv_path.c_str(), write_header ? "w" : "a");
  if (!f) errorQuda("Could not open aggregate CSV %s for write", csv_path.c_str());
  if (write_header) fputs(header, f);
  fprintf(f, "%s,%s,%d,%g,%g,%d,%.4f,%s,%s,%s,%s,%s,%s,%s\n", iso_timestamp().c_str(), baseline_model_name.c_str(), L,
          baseline_beta, kappa, baseline_rng, avg_iters, warm_s.c_str(), setup_s.c_str(), total_s.c_str(),
          rhs_s.c_str(), total_test_str.c_str(), resid_s.c_str(), conv_s.c_str());
  fclose(f);
}

// y += a * x
static inline void axpy_(double a, ColorSpinorField &x, ColorSpinorField &y)
{
  quda::blas::axpy(quda::cvector<double>(a), quda::cvector_ref<const ColorSpinorField> {x},
                   quda::cvector_ref<ColorSpinorField> {y});
}
// y = x + a * y
static inline void xpay_(ColorSpinorField &x, double a, ColorSpinorField &y)
{
  quda::blas::xpay(quda::cvector_ref<const ColorSpinorField> {x}, quda::cvector<double>(a),
                   quda::cvector_ref<ColorSpinorField> {y});
}

int main(int argc, char **argv)
{
  // Build the MG hierarchy on the normal operator D^dag D with a matrix-free
  // Galerkin coarse operator (read once by libquda on the first MG build).
  setenv("QUDA_MG_NORMAL_GALERKIN", "1", 1);

  setQudaDefaultMgTestParams();

  auto app = make_app();
  add_multigrid_option_group(app);
  add_comms_option_group(app);

  app->add_option("--baseline-gauge-file", baseline_gauge_files,
                  "Gauge configuration binary in QUDA QDP even-odd order (repeatable; one per config)");
  app->add_option("--baseline-per-model-csv", baseline_per_model_csv,
                  "Per-model CSV path (one setup row + num-rhs solve rows per gauge)");
  app->add_option("--baseline-aggregate-csv", baseline_aggregate_csv,
                  "Shared aggregate CSV path (one averaged row appended per run)");
  app->add_option("--baseline-model-name", baseline_model_name, "Value written to the CSV 'model' column");
  app->add_option("--baseline-beta", baseline_beta, "Gauge coupling beta tag for the CSV");
  app->add_option("--baseline-rng", baseline_rng, "rng tag of the source .dat for the CSV");
  app->add_option("--baseline-sample-base", baseline_sample_base, "CSV 'sample' index of the first gauge config");
  app->add_option("--baseline-num-rhs", baseline_num_rhs, "Number of random RHS solved per gauge config");
  app->add_option("--baseline-seed", baseline_seed, "Seed for the random Gaussian sources");

  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  if (baseline_gauge_files.empty()) errorQuda("No --baseline-gauge-file provided");
  if (mg_levels != 2)
    errorQuda("Galerkin normal-operator MG supports two levels only (got --mg-levels %d)", mg_levels);

  // Force the operator configuration for the baseline.
  dslash_type = QUDA_WILSON_DSLASH;
  fermion_t_boundary = QUDA_ANTI_PERIODIC_T;
  inv_multigrid = true;
  // Build the MG hierarchy on the full normal operator A = D^dag D (not even-odd
  // preconditioned): one V-cycle approximates (D^dag D)^{-1} directly.
  solve_type = QUDA_DIRECT_SOLVE;
  matpc_type = QUDA_MATPC_EVEN_EVEN; // unused for a direct solve

  // Full-operator smoother on every level (the Galerkin coarse operator acts on
  // the full normal operator, so even-odd smoothing is not applicable).
  for (int i = 0; i < QUDA_MAX_MG_LEVEL; i++) smoother_solve_type[i] = QUDA_DIRECT_SOLVE;

  setQudaPrecisions();
  initComms(argc, argv, gridsize_from_cmdline);

  gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);

  inv_param = newQudaInvertParam();
  mg_param = newQudaMultigridParam();
  mg_inv_param = newQudaInvertParam();

  setQudaMgSolveTypes();
  setMultigridInvertParam(inv_param);
  mg_param.invert_param = &mg_inv_param;
  for (int i = 0; i < mg_levels; i++) {
    if (mg_eig[i]) {
      mg_eig_param[i] = newQudaEigParam();
      setMultigridEigParam(mg_eig_param[i], i);
      mg_param.eig_param[i] = &mg_eig_param[i];
    } else {
      mg_param.eig_param[i] = nullptr;
    }
  }
  setMultigridParam(mg_param);

  // The Galerkin coarse operator is P^dag (D^dag D) P, which does not match the
  // link-based D_c = P^dag D P that verify() checks; disable the check.
  mg_param.run_verify = QUDA_BOOLEAN_FALSE;

  // The outer solve is a hand-written PCG on A = D^dag D; configure inv_param so
  // the Dirac operators we build for A and the source are the full Wilson op.
  inv_param.solution_type = QUDA_MAT_SOLUTION;
  inv_param.solve_type = QUDA_DIRECT_SOLVE;
  inv_param.inv_type = QUDA_CG_INVERTER; // unused; created Dirac is the same regardless
  inv_param.dagger = QUDA_DAG_NO;

  setDims(gauge_param.X);
  const int L = gauge_param.X[0];

  gauge_.resize(4 * V * gauge_site_size * host_gauge_data_type_size);
  for (int i = 0; i < 4; i++) gauge[i] = gauge_.data() + i * V * gauge_site_size * host_gauge_data_type_size;

  initQuda(device_ordinal);

  printfQuda("Galerkin-AMG PCG baseline (A = D^dag D): %zu config(s), L=%d, kappa=%g, tol=%g, maxiter=%d, num_rhs=%d\n",
             baseline_gauge_files.size(), L, kappa, tol, niter, baseline_num_rhs);

  init_per_model_csv(baseline_per_model_csv);

  // Accumulators for the aggregate row: setup/warmup are per gauge config; the
  // solve metrics are averaged over every right-hand side across all configs.
  double sum_setup_s = 0.0, sum_warmup_s = 0.0;
  double sum_rhs_time = 0.0, sum_iters = 0.0, sum_resid = 0.0, sum_converged = 0.0;
  long n_rhs_total = 0;
  std::chrono::high_resolution_clock::time_point t_test_start{};
  bool t_test_started = false;

  for (size_t c = 0; c < baseline_gauge_files.size(); c++) {
    const int sample = baseline_sample_base + (int)c;
    printfQuda("\n=== config %zu/%zu: %s (sample=%d) ===\n", c + 1, baseline_gauge_files.size(),
               baseline_gauge_files[c].c_str(), sample);

    load_gauge_binary(baseline_gauge_files[c]);
    loadGaugeQuda(gauge.data(), &gauge_param);

    double plaq[3];
    plaqQuda(plaq);
    printfQuda("Plaquette = %.12e (spatial = %.12e, temporal = %.12e)\n", plaq[0], plaq[1], plaq[2]);

    // Diagnostic mode: AMG_NO_PRECOND skips the AMG hierarchy and runs plain CG
    // on A = D^dag D (M^{-1} = I). Validates the QUDA operator independent of MG.
    const bool no_precond = (getenv("AMG_NO_PRECOND") != nullptr);

    // Build the Galerkin normal-operator AMG preconditioner. This hierarchy
    // build is the algorithmic setup cost (analogous to the NN forward pass
    // producing U_tilde on the neural side); time it as setup_cost_s.
    void *mg_handle = nullptr;
    quda::MG *mg_ptr = nullptr;
    double setup_cost_s = 0.0;
    if (!no_precond) {
      auto tset0 = std::chrono::high_resolution_clock::now();
      mg_handle = newMultigridQuda(&mg_param);
      auto tset1 = std::chrono::high_resolution_clock::now();
      setup_cost_s = std::chrono::duration<double>(tset1 - tset0).count();
      auto *mgs = static_cast<quda::multigrid_solver *>(mg_handle);
      mg_ptr = mgs->mg;
      printfQuda("MG setup done: %g secs (wall %.6f s)\n", mg_param.invert_param->secs, setup_cost_s);
    } else {
      printfQuda("AMG_NO_PRECOND set: running plain CG (M^{-1} = I) for operator validation.\n");
    }

    // Full Wilson Dirac operators for A = D^dag D (double) and a sloppy (single)
    // copy used for the V-cycle quality probe.
    quda::Dirac *dirac = nullptr, *diracSloppy = nullptr, *diracPre = nullptr;
    quda::createDirac(dirac, diracSloppy, diracPre, inv_param, /*pc_solve=*/false);

    // Field templates: device, double precision (outer), single (preconditioner).
    ColorSpinorParam cpuParam;
    constructWilsonTestSpinorParam(&cpuParam, &inv_param, &gauge_param);
    ColorSpinorParam devParam(cpuParam, inv_param, QUDA_CUDA_FIELD_LOCATION);
    devParam.create = QUDA_NULL_FIELD_CREATE;
    devParam.setPrecision(QUDA_DOUBLE_PRECISION, QUDA_DOUBLE_PRECISION, true);

    ColorSpinorParam preParam(devParam);
    preParam.setPrecision(QUDA_SINGLE_PRECISION, QUDA_SINGLE_PRECISION, true);

    ColorSpinorField psi(devParam), b(devParam), x(devParam), r(devParam), p(devParam), Ap(devParam), z(devParam);
    ColorSpinorField zsave(devParam); // previous preconditioned residual (flexible PCG)
    // Single-precision scratch for the AMG V-cycle (QUDA only compiles MG in
    // single/half precision).
    ColorSpinorField sr(preParam), sz(preParam), st(preParam);

    // Preconditioner z = M^{-1} r ~ (D^dag D)^{-1} r: a single Galerkin
    // normal-operator V-cycle (built on D^dag D), applied in single precision.
    auto apply_Minv = [&](ColorSpinorField &z_out, ColorSpinorField &r_in) {
      if (no_precond) { z_out = r_in; return; } // plain CG diagnostic
      sr = r_in;          // double -> single
      (*mg_ptr)(sz, sr);  // single V-cycle ~ (D^dag D)^{-1} r
      z_out = sz;         // single -> double
    };

    // Random Gaussian source psi (reproducible per config).
    quda::RNG rng(psi, baseline_seed + (unsigned long)c);
    spinorNoise(psi, rng, QUDA_NOISE_GAUSS);

    // Warm up autotuning of the matvec and preconditioner outside the solve
    // window. This one-time QUDA autotuning is logged separately as warmup_s
    // (analogous to JAX JIT compilation) and excluded from setup_cost_s.
    auto twarm0 = std::chrono::high_resolution_clock::now();
    if (!t_test_started) {
      t_test_start = twarm0;
      t_test_started = true;
    }
    dirac->MdagM(Ap, psi);
    apply_Minv(z, psi);
    auto twarm1 = std::chrono::high_resolution_clock::now();
    double warmup_s = std::chrono::duration<double>(twarm1 - twarm0).count();

    append_setup_row(baseline_per_model_csv, L, sample, setup_cost_s, warmup_s);
    sum_setup_s += setup_cost_s;
    sum_warmup_s += warmup_s;

    // Diagnostic: how well does the V-cycle approximate (D^dag D)^{-1}?
    // Report ||(D^dag D)(mg r) - r|| / ||r|| for random r (single precision).
    if (!no_precond) {
      spinorNoise(sr, rng, QUDA_NOISE_GAUSS); // sr = random (single, full)
      (*mg_ptr)(sz, sr);                      // sz ~ (D^dag D)^{-1} sr
      diracSloppy->MdagM(st, sz);             // st = (D^dag D) sz (single)
      double rn = std::sqrt(quda::blas::norm2(sr));
      double en = std::sqrt(quda::blas::xmyNorm(sr, st)); // ||st - sr||
      printfQuda("V-cycle check: ||(D^dag D)(V r) - r|| / ||r|| = %.6e\n", en / rn);
    }

    // ---- PCG on A = D^dag D, mirroring MatrixPreNet solver.py --------------
    // The AMG hierarchy (mg_ptr) and Dirac operators are reused across every
    // right-hand side; solving multiple RHS is what amortizes the setup cost.
    const double eps = tol;
    for (int j = 0; j < baseline_num_rhs; j++) {
      // Fresh random Gaussian source per RHS (advances the per-config RNG).
      spinorNoise(psi, rng, QUDA_NOISE_GAUSS);

      auto t0 = std::chrono::high_resolution_clock::now();

      dirac->Mdag(b, psi); // b = D^dag psi
      const double b_norm = std::sqrt(quda::blas::norm2(b));

      quda::blas::zero(x);
      r = b; // r = b - A x, with x = 0
      apply_Minv(z, r); // z = M^{-1} r
      p = z;
      double rz = quda::blas::cDotProduct(r, z).real();
      double r_norm = std::sqrt(quda::blas::norm2(r));

      int iters = 0;
      double final_resid = r_norm / b_norm;

      for (int k = 0; k < niter; k++) {
        if (r_norm / b_norm < eps) break;

        dirac->MdagM(Ap, p);
        double pAp = quda::blas::cDotProduct(p, Ap).real();
        double alpha = rz / pAp;

        axpy_(alpha, p, x);                                  // x += alpha p
        double r2 = quda::blas::axpyNorm(-alpha, Ap, r);     // r -= alpha Ap; ||r||^2
        r_norm = std::sqrt(r2);

        iters += 1;
        final_resid = r_norm / b_norm;

        zsave = z;          // z_k (previous preconditioned residual)
        apply_Minv(z, r);   // z_{k+1} = M^{-1} r_{k+1}
        double rz_new = quda::blas::cDotProduct(r, z).real();
        double rz_save = quda::blas::cDotProduct(r, zsave).real();
        // Flexible (Polak-Ribiere) beta; reduces to standard PCG when M^{-1} is
        // a fixed symmetric linear operator.
        double beta = (rz_new - rz_save) / rz;
        xpay_(z, beta, p); // p = z + beta p
        rz = rz_new;
      }

      auto t1 = std::chrono::high_resolution_clock::now();
      double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
      bool converged = (iters > 0) && (final_resid < eps);

      printfQuda("Result: rhs=%d iters=%d elapsed_s=%.4f final_resid=%.6e converged=%s\n", j, iters, elapsed_s,
                 final_resid, converged ? "True" : "False");

      append_rhs_row(baseline_per_model_csv, L, sample, j, elapsed_s, iters, final_resid, converged);
      sum_rhs_time += elapsed_s;
      sum_iters += iters;
      sum_resid += final_resid;
      sum_converged += converged ? 1.0 : 0.0;
      n_rhs_total += 1;
    }

    delete dirac;
    delete diracSloppy;
    delete diracPre;
    if (mg_handle) destroyMultigridQuda(mg_handle);
  }

  const int n_configs = (int)baseline_gauge_files.size();
  const double avg_setup = n_configs > 0 ? sum_setup_s / n_configs : 0.0;
  const double avg_warmup = n_configs > 0 ? sum_warmup_s / n_configs : 0.0;
  const double avg_rhs_time = n_rhs_total > 0 ? sum_rhs_time / n_rhs_total : 0.0;
  const double avg_iters = n_rhs_total > 0 ? sum_iters / n_rhs_total : 0.0;
  const double avg_resid = n_rhs_total > 0 ? sum_resid / n_rhs_total : 0.0;
  const double frac_converged = n_rhs_total > 0 ? sum_converged / n_rhs_total : 0.0;
  const auto t_test_end = std::chrono::high_resolution_clock::now();
  const double total_test_s =
    t_test_started ? std::chrono::duration<double>(t_test_end - t_test_start).count() : 0.0;

  append_aggregate_csv(baseline_aggregate_csv, L, avg_iters, avg_setup, avg_rhs_time, avg_warmup, total_test_s,
                       avg_resid, frac_converged);

  printfQuda("\nWrote %d setup + %ld rhs row(s) to %s; appended aggregate row to %s\n", n_configs, n_rhs_total,
             baseline_per_model_csv.c_str(), baseline_aggregate_csv.c_str());

  freeGaugeQuda();
  endQuda();
  finalizeComms();
  return 0;
}
