import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from numba import njit, prange
import time

@njit(parallel=True)
def _matvec_kernel(mx, my, nx, ny, panel_len, mu, out):
    M = mx.shape[0]
    for i in prange(M):
        s = 0.0
        for j in range(M):
            if i == j:
                # self-interaction is exactly zero for constant panels
                continue
            dx = mx[i] - mx[j]
            dy = my[i] - my[j]
            r2 = dx*dx + dy*dy
            # (x_i - x_j) . n_j
            dot = dx*nx[j] + dy*ny[j]
            # K_ij = -(1/2pi) * dot/r2 * len_j
            s += (dot / r2) * panel_len[j]
        # (1/2 I + K)mu: +1/2*mu_i - (1/2pi)*sum_j
        out[i] = 0.5 * mu[i] - s / (2.0 * np.pi) * mu[i]

# Corrected matvec: the sum over j must use mu[j], not mu[i]
@njit(parallel=True)
def _matvec_kernel_fixed(mx, my, nx, ny, panel_len, mu, out):
    M = mx.shape[0]
    for i in prange(M):
        s = 0.0
        for j in range(M):
            if i == j:
                continue
            dx = mx[i] - mx[j]
            dy = my[i] - my[j]
            r2 = dx*dx + dy*dy
            dot = dx*nx[j] + dy*ny[j]
            s += (dot / r2) * panel_len[j] * mu[j]
        out[i] = 0.5 * mu[i] - s / (2.0 * np.pi)

def bem_setup(M, a, b):
    t0 = time.perf_counter()
    theta = np.linspace(0, 2*np.pi, M + 1)

    x_nodes = a * np.cos(theta)
    y_nodes = b * np.sin(theta)

    x1, y1 = x_nodes[:-1], y_nodes[:-1]
    x2, y2 = x_nodes[1:],  y_nodes[1:]

    mx = 0.5 * (x1 + x2)
    my = 0.5 * (y1 + y2)

    # Tangent vector
    tx = x2 - x1
    ty = y2 - y1
    panel_len = np.sqrt(tx**2 + ty**2)
    t_hat_x = tx / panel_len
    t_hat_y = ty / panel_len

    # Outward normal: rotate tangent 90 degrees clockwise for CCW traversal → outward
    # For CCW parametrization: n = (ty, -tx) / |t| is outward for interior domain
    nx =  t_hat_y
    ny = -t_hat_x

    # Verify orientation: for an ellipse traversed CCW, outward normal dot centroid > 0
    # The ellipse is traversed CCW (theta increases), so this should be fine.
    # Quick sanity: dot(n, midpoint) should be positive for outward on an ellipse centred at origin
    dot_check = nx * mx + ny * my
    if np.mean(dot_check) < 0:
        nx = -nx
        ny = -ny

    f = mx**2 - my**2  # Dirichlet BC: u = x^2 - y^2

    setup_time = time.perf_counter() - t0
    return mx, my, nx, ny, panel_len, f, setup_time

def bem_matvec(mu, mx, my, nx, ny, panel_len):
    out = np.zeros_like(mu)
    _matvec_kernel_fixed(mx, my, nx, ny, panel_len, mu, out)
    return out

@njit(parallel=True)
def _evaluate_chunk(ex, ey, mx, my, nx, ny, panel_len, mu, out):
    N  = ex.shape[0]
    M  = mx.shape[0]
    for i in prange(N):
        s = 0.0
        for j in range(M):
            dx  = ex[i] - mx[j]
            dy  = ey[i] - my[j]
            r2  = dx*dx + dy*dy
            dot = dx*nx[j] + dy*ny[j]
            s  += mu[j] * (dot / r2) * panel_len[j]
        out[i] = -s / (2.0 * np.pi)

def bem_evaluate_chunked(ex, ey, mx, my, nx, ny, panel_len, mu, chunk_size=4096):
    N      = len(ex)
    result = np.zeros(N)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        _evaluate_chunk(
            ex[start:end], ey[start:end],
            mx, my, nx, ny, panel_len, mu,
            result[start:end]
        )
    return result

def run_bem(M, a=2.0, b=1.0, grid_n=200):
    mx, my, nx, ny, panel_len, f, setup_time = bem_setup(M, a, b)

    iters = [0]
    def callback(r):
        iters[0] += 1

    op = LinearOperator(
        (M, M),
        matvec=lambda mu: bem_matvec(mu, mx, my, nx, ny, panel_len),
        dtype=np.float64,
    )

    t1 = time.perf_counter()
    mu, info = gmres(op, f, rtol=1e-10, restart=100, maxiter=500, callback=callback, callback_type = 'legacy')
    solve_time = time.perf_counter() - t1

    # Interior evaluation
    t2   = time.perf_counter()
    gx   = np.linspace(-a, a, grid_n)
    gy   = np.linspace(-b, b, grid_n)
    GX, GY = np.meshgrid(gx, gy)
    GXf, GYf = GX.ravel(), GY.ravel()
    mask = (GXf / a)**2 + (GYf / b)**2 < 0.95
    ex, ey = GXf[mask], GYf[mask]

    u_num = bem_evaluate_chunked(ex, ey, mx, my, nx, ny, panel_len, mu)
    u_ex  = ex**2 - ey**2

    rel_l2    = np.linalg.norm(u_num - u_ex) / np.linalg.norm(u_ex)
    eval_time = time.perf_counter() - t2
    total_time = setup_time + solve_time + eval_time

    return {
        'M':               M,
        'iterations':      iters[0],
        'setup_time':      setup_time,
        'solve_time':      solve_time,
        'eval_time':       eval_time,
        'total_time':      total_time,
        'relative_L2_error': rel_l2,
    }

if __name__ == '__main__':
    a, b   = 2.0, 1.0
    M_list = [4000, 8000, 16000, 32000, 64000]

    # Warm up Numba JIT with tiny problem
    print("Warming up Numba JIT...")
    run_bem(64, a, b, grid_n=20)
    print("JIT warm-up done.\n")

    hdr = f"{'M':>8}  {'Iters':>6}  {'Setup(s)':>9}  {'Solve(s)':>9}  {'Eval(s)':>8}  {'Total(s)':>9}  {'Rel L2 Err':>12}"
    print(hdr)
    print("-" * len(hdr))

    for M in M_list:
        r = run_bem(M, a, b)
        print(
            f"{r['M']:>8}  {r['iterations']:>6}  {r['setup_time']:>9.3f}  "
            f"{r['solve_time']:>9.3f}  {r['eval_time']:>8.3f}  "
            f"{r['total_time']:>9.3f}  {r['relative_L2_error']:>12.3e}"
        )