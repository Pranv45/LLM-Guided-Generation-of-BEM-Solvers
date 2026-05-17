import numpy as np
import scipy.sparse.linalg as spla
from numba import njit, prange
import time

def bem_setup(a, b, M):
    theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
    nodes = np.zeros((M, 2), dtype=np.float64)
    nodes[:, 0] = a * np.cos(theta)
    nodes[:, 1] = b * np.sin(theta)

    elements = np.zeros((M, 2), dtype=np.int32)
    normals = np.zeros((M, 2), dtype=np.float64)
    lengths = np.zeros(M, dtype=np.float64)
    c_coeff = np.zeros(M, dtype=np.float64)

    for i in range(M):
        elements[i, 0] = i
        elements[i, 1] = (i + 1) % M

        dx = nodes[(i + 1) % M, 0] - nodes[i, 0]
        dy = nodes[(i + 1) % M, 1] - nodes[i, 1]
        lengths[i] = np.hypot(dx, dy)
        normals[i, 0] = dy / lengths[i]
        normals[i, 1] = -dx / lengths[i]

    for i in range(M):
        v1x = nodes[i, 0] - nodes[i - 1, 0]
        v1y = nodes[i, 1] - nodes[i - 1, 1]
        v2x = nodes[(i + 1) % M, 0] - nodes[i, 0]
        v2y = nodes[(i + 1) % M, 1] - nodes[i, 1]

        z = v1x * v2y - v1y * v2x
        d = v1x * v2x + v1y * v2y
        d_theta = np.arctan2(z, d)
        alpha = np.pi - d_theta
        c_coeff[i] = alpha / (2.0 * np.pi)

    return nodes, elements, normals, lengths, c_coeff

@njit(parallel=True, fastmath=True)
def bem_matvec(sigma, nodes, elements, normals, lengths, c_coeff, q_wts, phi1, phi2):
    M = len(sigma)
    num_q = len(q_wts)
    out = np.zeros(M, dtype=np.float64)

    for i in prange(M):
        xi = nodes[i, 0]
        yi = nodes[i, 1]
        val = 0.0

        for j in range(M):
            if j == i or j == (i - 1 + M) % M:
                continue

            nA = elements[j, 0]
            nB = elements[j, 1]

            Ax = nodes[nA, 0]
            Ay = nodes[nA, 1]
            Bx = nodes[nB, 0]
            By = nodes[nB, 1]
            nx = normals[j, 0]
            ny = normals[j, 1]

            J = lengths[j] / 2.0
            sigA = sigma[nA]
            sigB = sigma[nB]

            for q in range(num_q):
                p1 = phi1[q]
                p2 = phi2[q]

                yx = Ax * p1 + Bx * p2
                yy = Ay * p1 + By * p2

                dx = xi - yx
                dy = yi - yy
                r2 = dx*dx + dy*dy

                dot = dx * nx + dy * ny
                K = -dot / (2.0 * np.pi * r2)

                sig_val = sigA * p1 + sigB * p2
                val += K * sig_val * J * q_wts[q]

        out[i] = c_coeff[i] * sigma[i] + val

    return out

@njit(parallel=True, fastmath=True)
def _eval_chunk_numba(sigma, nodes, elements, normals, lengths, q_wts, phi1, phi2, eval_X, eval_Y):
    N = len(eval_X)
    M = len(elements)
    num_q = len(q_wts)
    out = np.zeros(N, dtype=np.float64)

    for i in prange(N):
        xi = eval_X[i]
        yi = eval_Y[i]
        val = 0.0

        for j in range(M):
            nA = elements[j, 0]
            nB = elements[j, 1]

            Ax = nodes[nA, 0]
            Ay = nodes[nA, 1]
            Bx = nodes[nB, 0]
            By = nodes[nB, 1]
            nx = normals[j, 0]
            ny = normals[j, 1]

            J = lengths[j] / 2.0
            sigA = sigma[nA]
            sigB = sigma[nB]

            for q in range(num_q):
                p1 = phi1[q]
                p2 = phi2[q]

                yx = Ax * p1 + Bx * p2
                yy = Ay * p1 + By * p2

                dx = xi - yx
                dy = yi - yy
                r2 = dx*dx + dy*dy

                dot = dx * nx + dy * ny
                K = -dot / (2.0 * np.pi * r2)

                sig_val = sigA * p1 + sigB * p2
                val += K * sig_val * J * q_wts[q]

        out[i] = val

    return out

def bem_evaluate_chunked(sigma, nodes, elements, normals, lengths, q_wts, phi1, phi2, eval_X, eval_Y, chunk_size=5000):
    N = len(eval_X)
    out = np.zeros(N, dtype=np.float64)
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        out[i:end] = _eval_chunk_numba(sigma, nodes, elements, normals, lengths, q_wts, phi1, phi2, eval_X[i:end], eval_Y[i:end])
    return out

class GMRESCounter:
    def __init__(self):
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1

def run_bem(M, a, b, eval_X, eval_Y, u_exact, q_wts, phi1, phi2):
    t0 = time.perf_counter()
    nodes, elements, normals, lengths, c_coeff = bem_setup(a, b, M)
    f = nodes[:, 0]**2 - nodes[:, 1]**2
    t_setup = time.perf_counter() - t0

    t1 = time.perf_counter()
    counter = GMRESCounter()

    def matvec_wrapper(sigma_vec):
        return bem_matvec(sigma_vec, nodes, elements, normals, lengths, c_coeff, q_wts, phi1, phi2)

    A_op = spla.LinearOperator((M, M), matvec=matvec_wrapper, dtype=np.float64)

    sigma_sol, _ = spla.gmres(A_op, f, rtol=1e-8, callback=counter, callback_type='pr_norm')
    t_solve = time.perf_counter() - t1

    t2 = time.perf_counter()
    u_approx = bem_evaluate_chunked(sigma_sol, nodes, elements, normals, lengths, q_wts, phi1, phi2, eval_X, eval_Y)
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

    q_pts, q_wts = np.polynomial.legendre.leggauss(8)
    phi1 = 0.5 * (1.0 - q_pts)
    phi2 = 0.5 * (1.0 + q_pts)

    n_warm, e_warm, norm_warm, len_warm, c_warm = bem_setup(a, b, 10)
    bem_matvec(np.zeros(10), n_warm, e_warm, norm_warm, len_warm, c_warm, q_wts, phi1, phi2)
    _eval_chunk_numba(np.zeros(10), n_warm, e_warm, norm_warm, len_warm, q_wts, phi1, phi2, eval_X[:2], eval_Y[:2])

    M_values = [4000, 8000, 16000, 32000, 64000]

    print(f"{'M':>10} | {'Iters':>8} | {'Setup(s)':>10} | {'Solve(s)':>10} | {'Eval(s)':>10} | {'Total(s)':>10} | {'Rel L2 Error':>15}")
    print("-" * 87)

    for M in M_values:
        res = run_bem(M, a, b, eval_X, eval_Y, u_exact, q_wts, phi1, phi2)
        print(f"{res['M']:10d} | {res['iterations']:8d} | {res['setup_time']:10.4f} | {res['solve_time']:10.4f} | {res['eval_time']:10.4f} | {res['total_time']:10.4f} | {res['relative_L2_error']:15.4e}")

if __name__ == "__main__":
    main()