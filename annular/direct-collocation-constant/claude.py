# ================================================================
# 2D BEM Solver — Laplace on Annulus  (1 < r < 2)
# Direct collocation | Constant elements | Matrix-free GMRES
# Numba-accelerated assembly and matvec
#
# Exact solution:  u(r,θ) = (r³ + r⁻³) cos(3θ)
# Inner BC (r=1): Dirichlet  u = 2 cos(3θ)
# Outer BC (r=2): Neumann   ∂u/∂n = (189/16) cos(3θ)
#
# KEY IMPLEMENTATION NOTES:
#   - Use EXACT outward normal at each quadrature point (not midpoint approx)
#   - Analytical G self-term (log-singular quadrature splitting)
#   - H self-term (principal value) = 0 for smooth circular arcs
#   - Free term c(ξ) = 1/2 added to H diagonal
# ================================================================

import numpy as np
from scipy.special import roots_legendre
from scipy.sparse.linalg import LinearOperator, gmres
import numba as nb
import time

# ----------------------------------------------------------------
# Exact solution
# ----------------------------------------------------------------

def u_exact(r, theta):
    return (r**3 + r**-3) * np.cos(3 * theta)

def dudr_exact(r, theta):
    """Radial derivative of u_exact."""
    return (3 * r**2 - 3 * r**-4) * np.cos(3 * theta)

# ----------------------------------------------------------------
# Quadrature
# ----------------------------------------------------------------

def gauss_quad(order=10):
    xi, wi = roots_legendre(order)
    return xi.astype(np.float64), wi.astype(np.float64)

# ----------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------

def build_geometry(N_inner, N_outer):
    """
    Constant-element discretisation of annulus boundaries.

    Returns
    -------
    midpoints  : (N,2)  Cartesian coords of element midpoints
    theta_mid  : (N,)   Angular midpoints
    radius_el  : (N,)   Radius of each element's circle
    ha         : (N,)   Half arc-angle  (= dθ/2)
    jac_el     : (N,)   Jacobian ds/dξ = r * ha
    bc         : (N,)   0 = Dirichlet (inner), 1 = Neumann (outer)

    Normal convention
    -----------------
    Inner boundary (r=1): outward from domain → −r̂  (radially inward)
    Outer boundary (r=2): outward from domain → +r̂  (radially outward)
    Normal is evaluated EXACTLY at each quadrature point (not midpoint approx).
    """
    dth_i = 2 * np.pi / N_inner
    ti    = np.arange(N_inner) * dth_i + dth_i / 2   # midpoint angles

    dth_o = 2 * np.pi / N_outer
    to    = np.arange(N_outer) * dth_o + dth_o / 2

    theta_mid = np.concatenate([ti, to])
    radius_el = np.concatenate([np.ones(N_inner), 2.0 * np.ones(N_outer)])

    midpoints = np.vstack([
        np.stack([np.cos(ti), np.sin(ti)], axis=1),
        np.stack([2 * np.cos(to), 2 * np.sin(to)], axis=1)
    ])

    # Both circles are discretised with the same angular span per element
    ha     = np.concatenate([np.full(N_inner, dth_i / 2),
                              np.full(N_outer, dth_o / 2)])
    jac_el = radius_el * ha   # ds = r * dθ_arc = r * ha * dξ

    bc = np.array([0] * N_inner + [1] * N_outer, dtype=np.int32)
    return midpoints, theta_mid, radius_el, ha, jac_el, bc

# ----------------------------------------------------------------
# Numba kernel: build H and G matrices
# ----------------------------------------------------------------

@nb.njit(cache=True, parallel=True, fastmath=True)
def _build_HG(midpoints, theta_mid, radius_el, ha, jac_el, xi_g, wi_g, H, Gm):
    """
    H[i,j]  = ∫_Γj  ∂G/∂n_y(x_i, y) ds_y   (no free term yet)
    Gm[i,j] = ∫_Γj  G(x_i, y) ds_y          (no self-term yet)

    Normal n(y) is the EXACT outward unit normal at each quadrature point y.
    G(x,y) = −1/(2π) log|x−y|,   ∂G/∂n_y = (1/2π)(x−y)·n / |x−y|²
    """
    N      = midpoints.shape[0]
    nq     = xi_g.shape[0]
    INV2PI = 1.0 / (2.0 * np.pi)

    for j in nb.prange(N):
        th_j = theta_mid[j]
        r_j  = radius_el[j]
        a    = ha[j]
        jac  = jac_el[j]
        # Outward-from-domain normal sign
        sign = -1.0 if r_j < 1.5 else 1.0

        for k in range(nq):
            tq   = th_j + xi_g[k] * a
            yx   = r_j * np.cos(tq)
            yy_  = r_j * np.sin(tq)
            # Exact outward normal at quadrature point y
            nx_q = sign * np.cos(tq)
            ny_q = sign * np.sin(tq)
            w    = wi_g[k] * jac

            for i in range(N):
                rx = midpoints[i, 0] - yx
                ry = midpoints[i, 1] - yy_
                d2 = rx * rx + ry * ry
                if d2 < 1e-60:
                    continue   # self-term handled analytically below
                Gm[i, j] += -INV2PI * 0.5 * np.log(d2) * w
                H[i, j]  +=  INV2PI * (rx * nx_q + ry * ny_q) / d2 * w


@nb.njit(cache=True, fastmath=True)
def _g_self_diag(radius_el, ha, xi_g, wi_g, Gm):
    """
    Replace the (inaccurate) standard-Gauss self-term Gm[j,j] with the
    analytically regularised value.

    G_self = −r/(2π) ∫_{-a}^{a} log(2r|sin(t/2)|) dt

    Splitting: log(sin(t/2)) = log(t/2) + log(sin(t/2)/(t/2))
      Singular part:  2a(log(a/2) − 1)   (exact)
      Smooth part:    Gauss quadrature on [0, a]
    """
    N      = radius_el.shape[0]
    INV2PI = 1.0 / (2.0 * np.pi)
    nq     = xi_g.shape[0]

    for j in range(N):
        r_j = radius_el[j]
        a   = ha[j]
        js  = 0.5 * a        # Jacobian for [0, a] integration
        smooth = 0.0
        for k in range(nq):
            ts = 0.5 * a * (xi_g[k] + 1.0)   # t ∈ [0, a]
            if ts > 1e-15:
                ratio   = np.sin(ts / 2.0) / (ts / 2.0)
                smooth += wi_g[k] * np.log(ratio)
        smooth *= 2.0 * js
        log_a2  = np.log(a / 2.0) if a > 1e-300 else 0.0
        Gm[j, j] = -r_j * INV2PI * (
            2.0 * a * np.log(2.0 * r_j)
            + 2.0 * a * (log_a2 - 1.0)
            + smooth
        )

# ----------------------------------------------------------------
# Numba matvec (for matrix-free GMRES)
# ----------------------------------------------------------------

@nb.njit(cache=True, parallel=True, fastmath=True)
def _matvec_nb(x, A_mat, out):
    N = A_mat.shape[0]
    for i in nb.prange(N):
        s = 0.0
        for j in range(N):
            s += A_mat[i, j] * x[j]
        out[i] = s

# ----------------------------------------------------------------
# Numba interior evaluation
# ----------------------------------------------------------------

@nb.njit(cache=True, parallel=True, fastmath=True)
def _eval_interior(pts_x, pts_y,
                   theta_mid, radius_el, ha, jac_el,
                   u_all, q_all, xi_g, wi_g, u_int):
    """
    Representation formula for interior points:
        u(x) = ∫G(x,y)q(y)ds_y − ∫∂G/∂n_y(x,y)u(y)ds_y
    """
    M      = pts_x.shape[0]
    N      = theta_mid.shape[0]
    nq     = xi_g.shape[0]
    INV2PI = 1.0 / (2.0 * np.pi)

    for ip in nb.prange(M):
        cx  = pts_x[ip];  cy  = pts_y[ip]
        val = 0.0
        for j in range(N):
            th_j = theta_mid[j]; r_j = radius_el[j]
            a    = ha[j];        jac = jac_el[j]
            sign = -1.0 if r_j < 1.5 else 1.0
            for k in range(nq):
                tq   = th_j + xi_g[k] * a
                yx   = r_j * np.cos(tq)
                yy_  = r_j * np.sin(tq)
                nx_q = sign * np.cos(tq)
                ny_q = sign * np.sin(tq)
                rx   = cx - yx;  ry = cy - yy_
                d2   = rx * rx + ry * ry
                if d2 < 1e-60:
                    continue
                w    = wi_g[k] * jac
                G_k  = -INV2PI * 0.5 * np.log(d2)
                H_k  =  INV2PI * (rx * nx_q + ry * ny_q) / d2
                val += (G_k * q_all[j] - H_k * u_all[j]) * w
        u_int[ip] = val

# ----------------------------------------------------------------
# Main BEM solver
# ----------------------------------------------------------------

def solve_bem(N, xi_g, wi_g, verbose=True):
    """
    Solve Laplace BVP on annulus 1<r<2 using N constant elements
    on each boundary (N_inner = N_outer = N).
    """
    N_inner = N_outer = N
    Ntot = 2 * N

    t0 = time.perf_counter()

    # --- Geometry ---
    midpoints, theta_mid, radius_el, ha, jac_el, bc = \
        build_geometry(N_inner, N_outer)
    ti = theta_mid[:N_inner]   # inner midpoint angles
    to = theta_mid[N_inner:]   # outer midpoint angles

    # --- Known boundary data ---
    # Inner (bc=0, Dirichlet): u = u_exact(1, θ);  q = −du/dr|_{r=1} = 0
    # Outer (bc=1, Neumann):   q = du/dr|_{r=2};   u is unknown
    u_bc = np.zeros(Ntot); q_bc = np.zeros(Ntot)
    u_bc[:N_inner] = u_exact(1.0, ti)
    q_bc[N_inner:] = dudr_exact(2.0, to)   # n = +r̂ at outer → q = du/dr

    # --- Build H and G matrices ---
    H  = np.zeros((Ntot, Ntot))
    Gm = np.zeros((Ntot, Ntot))
    _build_HG(midpoints, theta_mid, radius_el, ha, jac_el, xi_g, wi_g, H, Gm)
    _g_self_diag(radius_el, ha, xi_g, wi_g, Gm)   # fix G diagonal

    # Add free term c = 1/2 to H diagonal (H_self principal value ≈ 0)
    for i in range(Ntot):
        H[i, i] += 0.5

    t_setup = time.perf_counter() - t0

    # --- Assemble linear system ---
    # BIE: H @ u = Gm @ q
    # Unknown vector x: [q_inner (N), u_outer (N)]
    #
    # For Dirichlet col j (bc=0): u_j known  →  move H[:,j]*u_j to rhs
    #                                         →  keep −Gm[:,j]*q_j on lhs
    # For Neumann  col j (bc=1): q_j known  →  move Gm[:,j]*q_j to rhs
    #                                         →  keep  H[:,j]*u_j on lhs
    A   = np.zeros((Ntot, Ntot))
    rhs = np.zeros(Ntot)

    for j in range(Ntot):
        if bc[j] == 0:            # Dirichlet: q unknown, u known
            A[:, j]  = -Gm[:, j]
            rhs     -=  H[:, j] * u_bc[j]
        else:                     # Neumann: u unknown, q known
            A[:, j]  =  H[:, j]
            rhs     +=  Gm[:, j] * q_bc[j]

    # --- Matrix-free GMRES ---
    iters  = [0]
    out_buf = np.zeros(Ntot)

    def matvec(x):
        iters[0] += 1
        _matvec_nb(x, A, out_buf)
        return out_buf.copy()

    Aop = LinearOperator((Ntot, Ntot), matvec=matvec, dtype=np.float64)

    t1 = time.perf_counter()
    x_sol, info = gmres(Aop, rhs, rtol=1e-10, atol=1e-12,
                        restart=200, maxiter=2000, callback_type= 'legacy')
    t_solve = time.perf_counter() - t1

    if verbose:
        print(f"    GMRES: info={info}, matvecs={iters[0]}")

    # --- Reconstruct full boundary solution ---
    u_all = u_bc.copy(); q_all = q_bc.copy()
    q_all[:N_inner] = x_sol[:N_inner]   # solved q on inner
    u_all[N_inner:] = x_sol[N_inner:]   # solved u on outer

    # --- Interior evaluation ---
    t2 = time.perf_counter()

    ngrid = 60
    xg = np.linspace(-1.9, 1.9, ngrid)
    yg = np.linspace(-1.9, 1.9, ngrid)
    XX, YY = np.meshgrid(xg, yg)
    rr   = np.sqrt(XX**2 + YY**2)
    mask = (rr > 1.0) & (rr < 2.0)
    pts_x = XX[mask].astype(np.float64)
    pts_y = YY[mask].astype(np.float64)
    u_int = np.zeros(len(pts_x))

    _eval_interior(pts_x, pts_y,
                   theta_mid, radius_el, ha, jac_el,
                   u_all, q_all, xi_g, wi_g, u_int)

    t_eval = time.perf_counter() - t2

    # --- Error ---
    u_ref  = u_exact(np.sqrt(pts_x**2 + pts_y**2), np.arctan2(pts_y, pts_x))
    rel_l2 = np.linalg.norm(u_int - u_ref) / np.linalg.norm(u_ref)
    t_tot  = time.perf_counter() - t0

    return dict(N=N, unknowns=Ntot, iters=iters[0], rel_l2=rel_l2,
                t_setup=t_setup, t_solve=t_solve, t_eval=t_eval, t_total=t_tot)

# ----------------------------------------------------------------
# Warmup JIT
# ----------------------------------------------------------------

def warmup(xi_g, wi_g):
    solve_bem(8, xi_g, wi_g, verbose=False)

# ----------------------------------------------------------------
# Run refinement study
# ----------------------------------------------------------------

xi_g, wi_g = gauss_quad(order=10)

print("Warming up Numba JIT...")
warmup(xi_g, wi_g)
print("Warmup done.\n")

N_values = [160, 320, 640, 1280, 2560]

hdr = (f"{'N':>6} {'Unknowns':>10} {'GMRES Mvecs':>12} {'Rel L2 Err':>14} "
       f"{'Conv Rate':>10} {'Setup(s)':>10} {'Solve(s)':>10} {'Eval(s)':>10} {'Total(s)':>10}")
print(hdr)
print("-" * len(hdr))

results = []
for N in N_values:
    print(f"  Running N={N} ...")
    res = solve_bem(N, xi_g, wi_g, verbose=True)
    results.append(res)

    if len(results) > 1:
        prev = results[-2]
        rate = np.log2(prev['rel_l2'] / res['rel_l2']) / np.log2(res['N'] / prev['N'])
        rate_str = f"{rate:>10.2f}"
    else:
        rate_str = f"{'—':>10}"

    print(f"{res['N']:>6} {res['unknowns']:>10} {res['iters']:>12} "
          f"{res['rel_l2']:>14.6e} {rate_str} {res['t_setup']:>10.3f} "
          f"{res['t_solve']:>10.3f} {res['t_eval']:>10.3f} {res['t_total']:>10.3f}")
    print()

# Overall convergence order: least-squares fit of log(err) vs log(N)
Ns   = np.array([r['N'] for r in results], dtype=float)
errs = np.array([r['rel_l2'] for r in results])
coeffs = np.polyfit(np.log2(Ns), np.log2(errs), 1)
print(f"Overall convergence order (least-squares fit): {-coeffs[0]:.3f}")
print(f"  (theoretical: ~2.0 for constant elements on smooth domains)")