import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from numba import njit, prange
import time

@njit(parallel=True)
def _matvec_kernel(cx, cy, nx_el, ny_el, ex1, ey1, ex2, ey2, el_len, sigma, out, xi_g, w_g):
    M = cx.shape[0]
    Ng = xi_g.shape[0]
    for i in prange(M):
        s = 0.0
        for e in range(M):
            # shape function values at quadrature points: phi1=(1-xi)/2, phi2=(1+xi)/2
            # geometry: x(xi) = phi1*x1 + phi2*x2, jacobian = len/2
            jac = el_len[e] * 0.5
            nxe = nx_el[e]
            nye = ny_el[e]
            s1 = sigma[e]          # node e
            s2 = sigma[(e+1) % M]  # node e+1
            for g in range(Ng):
                xi = xi_g[g]
                w  = w_g[g]
                phi1 = 0.5 * (1.0 - xi)
                phi2 = 0.5 * (1.0 + xi)
                qx = phi1 * ex1[e] + phi2 * ex2[e]
                qy = phi1 * ey1[e] + phi2 * ey2[e]
                dx = cx[i] - qx
                dy = cy[i] - qy
                r2 = dx*dx + dy*dy
                if r2 < 1e-28:
                    continue
                dot = dx*nxe + dy*nye
                sig_q = phi1*s1 + phi2*s2
                s += w * (dot / r2) * sig_q * jac
        out[i] = 0.5 * sigma[i] - s / (2.0 * np.pi)


@njit(parallel=True)
def _evaluate_kernel(ex, ey, nx_el, ny_el, bx1, by1, bx2, by2, el_len, sigma, out, xi_g, w_g, M):
    N  = ex.shape[0]
    Ng = xi_g.shape[0]
    for i in prange(N):
        s = 0.0
        for e in range(M):
            jac = el_len[e] * 0.5
            nxe = nx_el[e]
            nye = ny_el[e]
            s1  = sigma[e]
            s2  = sigma[(e+1) % M]
            for g in range(Ng):
                xi   = xi_g[g]
                w    = w_g[g]
                phi1 = 0.5 * (1.0 - xi)
                phi2 = 0.5 * (1.0 + xi)
                qx   = phi1 * bx1[e] + phi2 * bx2[e]
                qy   = phi1 * by1[e] + phi2 * by2[e]
                dx   = ex[i] - qx
                dy   = ey[i] - qy
                r2   = dx*dx + dy*dy
                if r2 < 1e-28:
                    continue
                dot  = dx*nxe + dy*nye
                sig_q = phi1*s1 + phi2*s2
                s   += w * (dot / r2) * sig_q * jac
        out[i] = -s / (2.0 * np.pi)


def bem_setup(M, a, b):
    t0 = time.perf_counter()

    # M nodes uniformly in theta
    theta = np.linspace(0.0, 2.0*np.pi, M, endpoint=False)
    xn = a * np.cos(theta)
    yn = b * np.sin(theta)

    # Elements: element e connects node e -> node (e+1)%M
    x1 = xn
    y1 = yn
    x2 = np.roll(xn, -1)
    y2 = np.roll(yn, -1)

    tx = x2 - x1
    ty = y2 - y1
    el_len = np.sqrt(tx**2 + ty**2)

    # Outward normal: rotate tangent (tx,ty) by -90 deg -> (ty, -tx), normalise
    nx_el =  ty / el_len
    ny_el = -tx / el_len

    # Verify outward: dot with midpoint centroid direction should be > 0
    mx_el = 0.5*(x1+x2)
    my_el = 0.5*(y1+y2)
    if np.mean(nx_el*mx_el + ny_el*my_el) < 0.0:
        nx_el = -nx_el
        ny_el = -ny_el

    # free-term c_i: for a smooth boundary = 0.5 everywhere
    c = np.full(M, 0.5)

    # RHS: Dirichlet BC at nodes
    f = xn**2 - yn**2

    # Gauss-Legendre quadrature on [-1,1]
    xi_g, w_g = np.polynomial.legendre.leggauss(8)
    xi_g = xi_g.astype(np.float64)
    w_g  = w_g.astype(np.float64)

    setup_time = time.perf_counter() - t0
    return (xn, yn, x1, y1, x2, y2, nx_el, ny_el, el_len, c, f,
            xi_g, w_g, setup_time)


def bem_matvec(sigma, xn, yn, nx_el, ny_el, x1, y1, x2, y2, el_len, xi_g, w_g):
    M   = xn.shape[0]
    out = np.zeros(M, dtype=np.float64)
    _matvec_kernel(xn, yn, nx_el, ny_el, x1, y1, x2, y2, el_len,
                   sigma, out, xi_g, w_g)
    return out


def bem_evaluate_chunked(ex, ey, nx_el, ny_el, x1, y1, x2, y2, el_len,
                          sigma, xi_g, w_g, chunk_size=2048):
    M      = x1.shape[0]
    N      = ex.shape[0]
    result = np.zeros(N, dtype=np.float64)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        tmp = np.zeros(end - start, dtype=np.float64)
        _evaluate_kernel(ex[start:end], ey[start:end],
                         nx_el, ny_el, x1, y1, x2, y2, el_len,
                         sigma, tmp, xi_g, w_g, M)
        result[start:end] = tmp
    return result


def run_bem(M, a=2.0, b=1.0, grid_n=200):
    (xn, yn, x1, y1, x2, y2, nx_el, ny_el, el_len, c, f,
     xi_g, w_g, setup_time) = bem_setup(M, a, b)

    iters = [0]
    def callback(r):
        iters[0] += 1

    op = LinearOperator(
        (M, M),
        matvec=lambda sigma: bem_matvec(
            sigma, xn, yn, nx_el, ny_el, x1, y1, x2, y2, el_len, xi_g, w_g),
        dtype=np.float64,
    )

    t1 = time.perf_counter()
    sigma, info = gmres(op, f, rtol=1e-10, restart=100, maxiter=500,
                        callback=callback, callback_type='pr_norm')
    solve_time = time.perf_counter() - t1

    # Interior evaluation grid
    t2  = time.perf_counter()
    gx  = np.linspace(-a, a, grid_n)
    gy  = np.linspace(-b, b, grid_n)
    GX, GY = np.meshgrid(gx, gy)
    GXf, GYf = GX.ravel(), GY.ravel()
    mask = (GXf/a)**2 + (GYf/b)**2 < 0.95
    ex_int, ey_int = GXf[mask], GYf[mask]

    u_num = bem_evaluate_chunked(ex_int, ey_int, nx_el, ny_el,
                                  x1, y1, x2, y2, el_len, sigma, xi_g, w_g)
    u_ex  = ex_int**2 - ey_int**2
    rel_l2 = np.linalg.norm(u_num - u_ex) / np.linalg.norm(u_ex)
    eval_time  = time.perf_counter() - t2
    total_time = setup_time + solve_time + eval_time

    return {
        'M':                 M,
        'iterations':        iters[0],
        'setup_time':        setup_time,
        'solve_time':        solve_time,
        'eval_time':         eval_time,
        'total_time':        total_time,
        'relative_L2_error': rel_l2,
    }


if __name__ == '__main__':
    a, b   = 2.0, 1.0
    M_list = [4000, 8000, 16000, 32000, 64000]

    print("Warming up Numba JIT ...")
    run_bem(64, a, b, grid_n=20)
    print("JIT warm-up done.\n")

    hdr = (f"{'M':>8}  {'Iters':>6}  {'Setup(s)':>9}  {'Solve(s)':>9}"
           f"  {'Eval(s)':>8}  {'Total(s)':>9}  {'Rel L2 Err':>12}")
    print(hdr)
    print("-" * len(hdr))

    for M in M_list:
        r = run_bem(M, a, b)
        print(f"{r['M']:>8}  {r['iterations']:>6}  {r['setup_time']:>9.3f}"
              f"  {r['solve_time']:>9.3f}  {r['eval_time']:>8.3f}"
              f"  {r['total_time']:>9.3f}  {r['relative_L2_error']:>12.3e}")