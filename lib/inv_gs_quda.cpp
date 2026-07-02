#include <invert_quda.h>
#include <blas_quda.h>
#include <color_spinor_field.h>
#include <util_quda.h>
#include <solver.hpp>

/**
   @file inv_gs_quda.cpp

   Red-black (checkerboard) Gauss-Seidel relaxation for use as a multigrid
   smoother on a Hermitian operator.  In QUDA a full field is stored as two
   checkerboards whose index is exactly the site parity (x+y+z+t) mod 2, so the
   "red" and "black" colors are simply the even and odd parity subfields and a
   colored sweep is a masked update of one subfield.  A sweep updates one
   checkerboard from the current residual, then the other.  A forward sweep
   (even before odd) is used as the pre-smoother and a backward sweep (odd before
   even) as the post-smoother.

   Two relaxation modes, selected by SolverParam::gs_linear:

   - gs_linear == true: a fixed number of sweeps with a fixed scalar step
     omega/lambda_max.  The coefficients do not depend on the right-hand side, so
     the map is linear and the forward/backward pre/post pair (matched sweep
     counts) is self-adjoint.  This is what keeps the AMG preconditioner
     Hermitian.

   - gs_linear == false (default): standard practice.  Each colored sweep uses
     the minimal-residual optimal step computed from the current residual, and
     the sweep loop exits early once the residual tolerance is met.  This is
     generally non-linear (input-dependent coefficients and stopping).
*/

namespace quda
{

  GaussSeidel::GaussSeidel(const DiracMatrix &mat, const DiracMatrix &matSloppy, SolverParam &param) :
    Solver(mat, matSloppy, matSloppy, matSloppy, param)
  {
    if (!mat.hermitian()) errorQuda("Gauss-Seidel smoother requires a Hermitian operator");
  }

  void GaussSeidel::create(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b)
  {
    Solver::create(x, b);

    if (!init || r.size() != b.size()) {
      resize(r, b.size(), QUDA_NULL_FIELD_CREATE, b[0]);
      // The non-linear path needs temporaries for the colored search direction
      // and its image under mat; the linear path does not.
      if (!param.gs_linear) {
        resize(d, b.size(), QUDA_NULL_FIELD_CREATE, b[0]);
        resize(Ad, b.size(), QUDA_NULL_FIELD_CREATE, b[0]);
      }
      init = true;
    }

    // Linear path only: estimate a scalar diagonal (the largest eigenvalue) once,
    // so the smoother is a fixed linear operator on every subsequent
    // application.  A step of omega / lambda_max is the standard stable
    // relaxation scale for a Jacobi / Gauss-Seidel smoother on a Hermitian
    // positive-definite operator.
    if (param.gs_linear && diag_inv == 0.0) {
      ColorSpinorParam csParam(b[0]);
      csParam.create = QUDA_NULL_FIELD_CREATE;
      ColorSpinorField t1(csParam), t2(csParam);
      double lambda_max = performPowerIterations(mat, b[0], t1, t2, 20, 1);
      if (!(lambda_max > 0.0)) errorQuda("Gauss-Seidel diagonal estimate is non-positive (%e)", lambda_max);
      diag_inv = 1.0 / lambda_max;
      logQuda(QUDA_VERBOSE, "Gauss-Seidel scalar diagonal estimate lambda_max = %e\n", lambda_max);
    }
  }

  cvector_ref<const ColorSpinorField> GaussSeidel::get_residual()
  {
    if (!init) errorQuda("No residual vector present");
    if (!param.return_residual) errorQuda("SolverParam::return_residual not enabled");
    return r;
  }

  void GaussSeidel::operator()(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b)
  {
    if (param.maxiter == 0) {
      if (param.use_init_guess == QUDA_USE_INIT_GUESS_NO) blas::zero(x);
      return;
    }

    create(x, b);

    if (!param.is_preconditioner) getProfile().TPSTART(QUDA_PROFILE_COMPUTE);

    if (param.use_init_guess == QUDA_USE_INIT_GUESS_NO) blas::zero(x);

    const double omega = (param.omega != 0.0 ? param.omega : 1.0);
    const double alpha_fixed = omega * diag_inv;

    // Convergence bookkeeping is only used on the non-linear (standard) path.
    auto b2 = param.gs_linear ? vector<double>(b.size(), 1.0) : blas::norm2(b);
    auto stop = param.gs_linear ? vector<double>(b.size(), 0.0) :
                                  stopping(param.tol, b2, param.residual_type);

    // A half-sweep updates a single checkerboard (parity) from the current
    // residual r = b - A x.  If the smoother operates on an even-odd
    // preconditioned (single-parity) field then the colored update degenerates
    // to an update of the whole field.
    auto half_sweep = [&](QudaParity parity) {
      const QudaParity other = (parity == QUDA_EVEN_PARITY) ? QUDA_ODD_PARITY : QUDA_EVEN_PARITY;
      mat(r, x);              // r = A x
      blas::xpay(b, -1.0, r); // r = b - A x
      for (auto i = 0u; i < b.size(); i++) {
        const bool full = (x[i].SiteSubset() == QUDA_FULL_SITE_SUBSET);
        if (param.gs_linear) {
          // Fixed scalar step: x_cb += (omega / lambda_max) r_cb
          if (full) {
            blas::axpy(alpha_fixed, r[i][parity], x[i][parity]);
          } else {
            blas::axpy(alpha_fixed, r[i], x[i]);
          }
        } else {
          // Standard practice: minimal-residual optimal step for the colored
          // direction d (= residual masked to this checkerboard).  The step
          //   a = omega <A d, r> / <A d, A d>
          // is computed from the current residual, so it is input-dependent.
          blas::copy(d[i], r[i]);
          if (full) blas::zero(d[i][other]); // restrict the direction to this color
          mat(Ad[i], d[i]);
          const double den = blas::norm2(Ad[i]);
          const double a = den > 0.0 ? omega * blas::reDotProduct(Ad[i], r[i]) / den : 0.0;
          if (full) {
            blas::axpy(a, r[i][parity], x[i][parity]);
          } else {
            blas::axpy(a, r[i], x[i]);
          }
        }
      }
    };

    int sweep = 0;
    for (; sweep < param.maxiter; sweep++) {
      if (param.gs_order == QUDA_GS_BACKWARD_ORDER) {
        half_sweep(QUDA_ODD_PARITY);
        half_sweep(QUDA_EVEN_PARITY);
      } else {
        half_sweep(QUDA_EVEN_PARITY);
        half_sweep(QUDA_ODD_PARITY);
      }

      // Standard practice: exit early once the residual tolerance is met.
      if (!param.gs_linear) {
        mat(r, x);
        auto r2 = blas::xmyNorm(b, r); // r = b - A x, returns |r|^2
        if (convergenceL2(r2, stop)) {
          sweep++;
          break;
        }
      }
    }

    if (param.return_residual) {
      mat(r, x);
      blas::xpay(b, -1.0, r); // r = b - A x
    }

    if (!param.is_preconditioner) {
      getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
      param.iter += sweep;
    }
  }

} // namespace quda
