#include <invert_quda.h>
#include <blas_quda.h>
#include <color_spinor_field.h>
#include <util_quda.h>
#include <solver.hpp>

/**
   @file inv_chebyshev_quda.cpp

   Fixed-degree Chebyshev iteration.  This applies a fixed polynomial p_K(A) to
   the source, approximating A^{-1} b over the spectral interval
   [lambda_min, lambda_max].  The polynomial coefficients depend only on the
   spectral bounds and the degree K = SolverParam::maxiter, and are independent
   of the right-hand side, so the map is linear and (for a Hermitian A)
   self-adjoint.  This makes it a symmetric replacement for the early-stopping CG
   coarse solve in the multigrid V-cycle.
*/

namespace quda
{

  ChebyshevIter::ChebyshevIter(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matEig,
                               SolverParam &param) :
    Solver(mat, matSloppy, matSloppy, matEig, param)
  {
    if (!mat.hermitian()) errorQuda("Chebyshev iteration requires a Hermitian operator");
  }

  void ChebyshevIter::create(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b)
  {
    Solver::create(x, b);

    if (!init || r.size() != b.size()) {
      resize(r, b.size(), QUDA_NULL_FIELD_CREATE, b[0]);
      resize(p, b.size(), QUDA_NULL_FIELD_CREATE, b[0]);
      init = true;
    }

    // Determine the spectral interval once (fixed for the life of the solver so
    // that the applied polynomial is a fixed linear operator).
    if (lambda_max == 0.0) {
      if (param.ca_lambda_max > 0.0) {
        lambda_max = param.ca_lambda_max;
      } else {
        ColorSpinorParam csParam(b[0]);
        csParam.create = QUDA_NULL_FIELD_CREATE;
        ColorSpinorField t1(csParam), t2(csParam);
        lambda_max = performPowerIterations(mat, b[0], t1, t2, 20, 1);
      }
      lambda_min = param.ca_lambda_min;
      if (!(lambda_max > lambda_min && lambda_min >= 0.0))
        errorQuda("Invalid Chebyshev spectral bounds [%e, %e]", lambda_min, lambda_max);
      logQuda(QUDA_VERBOSE, "Chebyshev coarse solve using spectral interval [%e, %e], degree %d\n", lambda_min,
              lambda_max, param.maxiter);
    }
  }

  cvector_ref<const ColorSpinorField> ChebyshevIter::get_residual()
  {
    if (!init) errorQuda("No residual vector present");
    if (!param.return_residual) errorQuda("SolverParam::return_residual not enabled");
    return r;
  }

  void ChebyshevIter::operator()(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b)
  {
    const int K = param.maxiter;
    if (K <= 0) {
      if (param.use_init_guess == QUDA_USE_INIT_GUESS_NO) blas::zero(x);
      return;
    }

    create(x, b);

    if (!param.is_preconditioner) getProfile().TPSTART(QUDA_PROFILE_COMPUTE);

    const double theta = 0.5 * (lambda_max + lambda_min);
    const double delta = 0.5 * (lambda_max - lambda_min);
    const double sigma1 = theta / delta;
    double rho = 1.0 / sigma1;

    // r = b - A x
    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      mat(r, x);
      blas::xpay(b, -1.0, r);
    } else {
      blas::zero(x);
      blas::copy(r, b);
    }

    // p = (1/theta) r ; x += p
    blas::axpbyz(1.0 / theta, r, 0.0, r, p);
    blas::xpy(p, x);

    for (int k = 1; k < K; k++) {
      mat(r, x);
      blas::xpay(b, -1.0, r); // r = b - A x

      double rho_new = 1.0 / (2.0 * sigma1 - rho);
      double c_p = rho * rho_new;
      double c_r = 2.0 * rho_new / delta;

      // p = c_p * p + c_r * r
      blas::axpby(c_r, r, c_p, p);
      blas::xpy(p, x); // x += p
      rho = rho_new;
    }

    if (param.return_residual) {
      mat(r, x);
      blas::xpay(b, -1.0, r); // r = b - A x
    }

    if (!param.is_preconditioner) {
      getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
      param.iter += K;
    }
  }

} // namespace quda
