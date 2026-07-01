// Materialize the Galerkin-AMG preconditioner M^{-1} ~ (D^dag D)^{-1} as a dense
// N x N complex matrix in MatrixPreNet / eigval-comp flat spinor layout.
//
// Gauge: QDP even-odd binary (convert_gauge.py).
// Spinor layout matches jax_model materialize_su3_operator:
//   index = Y*(Nc*Ns) + c*Ns + s,  Y = x + L*(y + L*(z + L*t)),  Nc=3, Ns=4.
//
// Output: raw complex128 row-major (C order) N*N matrix (.bin).
// Use scripts/bin_to_minv_npz.py to wrap as .npz for eig-compare.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <complex>
#include <chrono>
#include <cstdint>

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

namespace {

constexpr int Nc = 3;
constexpr int Ns = 4;

QudaGaugeParam gauge_param;
QudaInvertParam inv_param;
QudaMultigridParam mg_param;
QudaInvertParam mg_inv_param;
QudaEigParam mg_eig_param[QUDA_MAX_MG_LEVEL];

std::vector<char> gauge_;
std::array<void *, 4> gauge;

std::vector<std::string> baseline_gauge_files;
std::string materialize_out;
int baseline_config_index = 0;

void load_gauge_binary(const std::string &path)
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
  if (pos != end) errorQuda("Gauge file %s has %ld trailing bytes", path.c_str(), end - pos);
}

void mnet_to_quda_buffer(const std::complex<double> *mnet, std::complex<double> *quda_buf, int L)
{
  const int Vh = (L * L * L * L) / 2;
  for (int t = 0; t < L; t++) {
    for (int z = 0; z < L; z++) {
      for (int y = 0; y < L; y++) {
        for (int x = 0; x < L; x++) {
          const int Y = x + L * (y + L * (z + L * t));
          const int parity = (x + y + z + t) & 1;
          const int x_cb = Y / 2;
          for (int s = 0; s < Ns; s++) {
            for (int c = 0; c < Nc; c++) {
              const size_t mnet_idx = (size_t)Y * Nc * Ns + (size_t)c * Ns + s;
              const size_t quda_idx = (size_t)parity * Vh * Ns * Nc + (size_t)(x_cb * Ns + s) * Nc + c;
              quda_buf[quda_idx] = mnet[mnet_idx];
            }
          }
        }
      }
    }
  }
}

void quda_buffer_to_mnet(const std::complex<double> *quda_buf, std::complex<double> *mnet, int L)
{
  const int Vh = (L * L * L * L) / 2;
  for (int t = 0; t < L; t++) {
    for (int z = 0; z < L; z++) {
      for (int y = 0; y < L; y++) {
        for (int x = 0; x < L; x++) {
          const int Y = x + L * (y + L * (z + L * t));
          const int parity = (x + y + z + t) & 1;
          const int x_cb = Y / 2;
          for (int s = 0; s < Ns; s++) {
            for (int c = 0; c < Nc; c++) {
              const size_t mnet_idx = (size_t)Y * Nc * Ns + (size_t)c * Ns + s;
              const size_t quda_idx = (size_t)parity * Vh * Ns * Nc + (size_t)(x_cb * Ns + s) * Nc + c;
              mnet[mnet_idx] = quda_buf[quda_idx];
            }
          }
        }
      }
    }
  }
}

void write_raw_matrix(const std::string &path, const std::vector<std::complex<double>> &M, int n)
{
  std::ofstream out(path, std::ios::binary);
  if (!out) errorQuda("Could not open output %s", path.c_str());
  int64_t dim = n;
  out.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
  out.write(reinterpret_cast<const char *>(M.data()), (size_t)n * n * sizeof(std::complex<double>));
  if (!out) errorQuda("Short write on output %s", path.c_str());
}

void materialize_mg(quda::MG *mg_ptr, int L, int n, const std::string &out_path)
{
  ColorSpinorParam cpuParam;
  constructWilsonTestSpinorParam(&cpuParam, &inv_param, &gauge_param);
  cpuParam.create = QUDA_REFERENCE_FIELD_CREATE;
  cpuParam.location = QUDA_CPU_FIELD_LOCATION;
  cpuParam.setPrecision(QUDA_DOUBLE_PRECISION, QUDA_DOUBLE_PRECISION, true);

  ColorSpinorParam devParam(cpuParam, inv_param, QUDA_CUDA_FIELD_LOCATION);
  devParam.create = QUDA_NULL_FIELD_CREATE;
  devParam.setPrecision(QUDA_DOUBLE_PRECISION, QUDA_DOUBLE_PRECISION, true);

  ColorSpinorParam preParam(devParam);
  preParam.setPrecision(QUDA_SINGLE_PRECISION, QUDA_SINGLE_PRECISION, true);

  const size_t quda_sites = (size_t)V * Ns * Nc;
  std::vector<std::complex<double>> e_mnet(n), col_mnet(n);
  std::vector<std::complex<double>> e_quda(quda_sites), z_quda(quda_sites);
  std::vector<std::complex<double>> M((size_t)n * n);

  ColorSpinorField sr(preParam), sz(preParam);

  auto t0 = std::chrono::high_resolution_clock::now();

  for (int j = 0; j < n; j++) {
    std::fill(e_mnet.begin(), e_mnet.end(), std::complex<double>(0.0, 0.0));
    e_mnet[j] = std::complex<double>(1.0, 0.0);
    mnet_to_quda_buffer(e_mnet.data(), e_quda.data(), L);

    cpuParam.v = e_quda.data();
    ColorSpinorField h_in(cpuParam);
    ColorSpinorField d_in(devParam);
    d_in = h_in;
    sr = d_in;
    (*mg_ptr)(sz, sr);

    ColorSpinorParam outCpuParam(cpuParam);
    outCpuParam.v = z_quda.data();
    ColorSpinorField h_out(outCpuParam);
    h_out = sz;
    quda_buffer_to_mnet(z_quda.data(), col_mnet.data(), L);

    for (int i = 0; i < n; i++) M[(size_t)i * n + j] = col_mnet[i];

    if ((j + 1) % 256 == 0 || j + 1 == n)
      printfQuda("  materialized column %d / %d\n", j + 1, n);
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  const double elapsed = std::chrono::duration<double>(t1 - t0).count();
  printfQuda("Materialization done in %.2f s\n", elapsed);

  write_raw_matrix(out_path, M, n);
  printfQuda("Wrote Minv (%d x %d) to %s\n", n, n, out_path.c_str());
}

} // namespace

int main(int argc, char **argv)
{
  setenv("QUDA_MG_NORMAL_GALERKIN", "1", 1);
  setQudaDefaultMgTestParams();

  auto app = make_app();
  add_multigrid_option_group(app);
  add_comms_option_group(app);

  app->add_option("--baseline-gauge-file", baseline_gauge_files,
                  "Gauge binary in QUDA QDP even-odd order (one per config)");
  app->add_option("--materialize-out", materialize_out, "Output raw .bin path (int64 n + complex128 data)")->required();
  app->add_option("--baseline-config-index", baseline_config_index,
                  "Config index tag written into logs (default 0)");

  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  if (baseline_gauge_files.empty()) errorQuda("No --baseline-gauge-file provided");
  if (mg_levels != 2) errorQuda("Galerkin normal-operator MG supports two levels only (got %d)", mg_levels);

  dslash_type = QUDA_WILSON_DSLASH;
  fermion_t_boundary = QUDA_ANTI_PERIODIC_T;
  inv_multigrid = true;
  solve_type = QUDA_DIRECT_SOLVE;
  matpc_type = QUDA_MATPC_EVEN_EVEN;
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
  mg_param.run_verify = QUDA_BOOLEAN_FALSE;

  inv_param.solution_type = QUDA_MAT_SOLUTION;
  inv_param.solve_type = QUDA_DIRECT_SOLVE;
  inv_param.inv_type = QUDA_CG_INVERTER;
  inv_param.dagger = QUDA_DAG_NO;

  setDims(gauge_param.X);
  const int L = gauge_param.X[0];
  const int n = Nc * Ns * L * L * L * L;

  gauge_.resize(4 * V * gauge_site_size * host_gauge_data_type_size);
  for (int i = 0; i < 4; i++) gauge[i] = gauge_.data() + i * V * gauge_site_size * host_gauge_data_type_size;

  initQuda(device_ordinal);

  printfQuda("AMG Minv materialize: %zu config(s), L=%d, N=%d, kappa=%g\n", baseline_gauge_files.size(), L, n,
             kappa);

  for (size_t c = 0; c < baseline_gauge_files.size(); c++) {
    printfQuda("\n=== config %zu/%zu: %s (cfg index tag=%d) ===\n", c + 1, baseline_gauge_files.size(),
               baseline_gauge_files[c].c_str(), baseline_config_index + (int)c);

    load_gauge_binary(baseline_gauge_files[c]);
    loadGaugeQuda(gauge.data(), &gauge_param);

    double plaq[3];
    plaqQuda(plaq);
    printfQuda("Plaquette = %.12e\n", plaq[0]);

    void *mg_handle = newMultigridQuda(&mg_param);
    auto *mgs = static_cast<quda::multigrid_solver *>(mg_handle);
    quda::MG *mg_ptr = mgs->mg;
    printfQuda("MG setup done: %g secs\n", mg_param.invert_param->secs);

    std::string out_path = materialize_out;
    if (baseline_gauge_files.size() > 1) {
      const auto dot = out_path.find_last_of('.');
      const int cfg = baseline_config_index + (int)c;
      if (dot == std::string::npos) {
        out_path += "_cfg" + std::to_string(cfg);
      } else {
        out_path = out_path.substr(0, dot) + "_cfg" + std::to_string(cfg) + out_path.substr(dot);
      }
    }

    materialize_mg(mg_ptr, L, n, out_path);
    destroyMultigridQuda(mg_handle);
  }

  freeGaugeQuda();
  endQuda();
  finalizeComms();
  return 0;
}
