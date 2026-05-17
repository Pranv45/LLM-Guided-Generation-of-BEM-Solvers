import numpy as np
import scipy.sparse.linalg as spla
from numba import njit, prange
import time

def bem_setup(a, b, M):
    theta = np.linspace(0, 2 * np.pi, M + 1)
    x_nodes = a * np.cos(theta)
    y_nodes = b * np.sin(theta)

    midpoints = np.zeros((M, 2), dtype=np.float64)
    normals = np.zeros((M, 2), dtype=np.float64)
    lengths = np.zeros(M, dtype=np.float64)

    for i in range(M):
        dx = x_nodes[i+1] - x_nodes[i]
        dy = y_nodes[i+1] - y_nodes[i]

        midpoints[i, 0] = x_nodes[i] + dx / 2.0
        midpoints[i, 1] = y_nodes[i] + dy / 2.0

        lengths[i] = np.hypot(dx, dy)

        normals[i, 0] = dy / lengths[i]
        normals[i, 1] = -dx / lengths[i]

    return midpoints, normals, lengths

@njit(parallel=True, fastmath=True)
def bem_matvec(mu, midpoints, normals, lengths):
    M = len(mu)
    out = np.zeros(M, dtype=np.float64)

    for i in prange(M):
        xi = midpoints[i, 0]
        yi = midpoints[i, 1]
        val = 0.0
        for j in range(M):
            if i == j:
                continue
            dx = xi - midpoints[j, 0]
            dy = yi - midpoints[j, 1]
            r2 = dx*dx + dy*dy
            dot = dx * normals[j, 0] + dy * normals[j, 1]
            val += - (dot / (2.0 * np.pi * r2)) * lengths[j] * mu[j]

        out[i] = 0.5 * mu[i] + val

    return out

def bem_evaluate_chunked(mu, midpoints, normals, lengths, eval_X, eval_Y, chunk_size=1000):
    N = len(eval_X)
    out = np.zeros(N, dtype=np.float64)

    mu_scaled = -mu * lengths / (2.0 * np.pi)

    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)

        X_chunk = eval_X[i:end, None]
        Y_chunk = eval_Y[i:end, None]

        dx = X_chunk - midpoints[:, 0]
        dy = Y_chunk - midpoints[:, 1]

        r2 = dx**2 + dy**2
        dot = dx * normals[:, 0] + dy * normals[:, 1]

        kernel = dot / r2
        out[i:end] = np.dot(kernel, mu_scaled)

    return out

class GMRESCounter:
    def __init__(self):
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1

def run_bem(M, a, b, eval_X, eval_Y, u_exact):
    t0 = time.perf_counter()
    midpoints, normals, lengths = bem_setup(a, b, M)
    f = midpoints[:, 0]**2 - midpoints[:, 1]**2
    t_setup = time.perf_counter() - t0

    t1 = time.perf_counter()
    counter = GMRESCounter()

    def matvec_wrapper(mu_vec):
        return bem_matvec(mu_vec, midpoints, normals, lengths)

    A_op = spla.LinearOperator((M, M), matvec=matvec_wrapper, dtype=np.float64)

    mu_sol, _ = spla.gmres(A_op, f, rtol=1e-8, callback=counter, callback_type= 'legacy')
    t_solve = time.perf_counter() - t1

    t2 = time.perf_counter()
    u_approx = bem_evaluate_chunked(mu_sol, midpoints, normals, lengths, eval_X, eval_Y)
    t_eval = time.perf_counter() - t2

    rel_l2_error = np.linalg.norm(u_approx - u_exact) / np.linalg.norm(u_exact)
    t_total = t_setup + t_solve + t_eval

    return {
        'M': M,
        'iterations': counter.niter,
        'setup_time': t_setup,
        'solve_time': t_solve,
        'eval_time': t_eval,
        'total_time': t_total,
        'relative_L2_error': rel_l2_error
    }

def main():
    a, b = 2.0, 1.0

    x_grid = np.linspace(-a, a, 100)
    y_grid = np.linspace(-b, b, 100)
    XX, YY = np.meshgrid(x_grid, y_grid)
    X_flat = XX.ravel()
    Y_flat = YY.ravel()

    mask = (X_flat/a)**2 + (Y_flat/b)**2 < 0.95
    eval_X = X_flat[mask]
    eval_Y = Y_flat[mask]
    u_exact = eval_X**2 - eval_Y**2

    mid_warm, norm_warm, len_warm = bem_setup(a, b, 10)
    bem_matvec(np.zeros(10), mid_warm, norm_warm, len_warm)

    M_values = [4000, 8000, 16000, 32000, 64000]

    print(f"{'M':>10} | {'Iters':>8} | {'Setup(s)':>10} | {'Solve(s)':>10} | {'Eval(s)':>10} | {'Total(s)':>10} | {'Rel L2 Error':>15}")
    print("-" * 87)

    for M in M_values:
        res = run_bem(M, a, b, eval_X, eval_Y, u_exact)
        print(f"{res['M']:10d} | {res['iterations']:8d} | {res['setup_time']:10.4f} | {res['solve_time']:10.4f} | {res['eval_time']:10.4f} | {res['total_time']:10.4f} | {res['relative_L2_error']:15.4e}")

if __name__ == "__main__":
    main()