#it 1
import numpy as np
import scipy.sparse.linalg as spla
from numba import njit, prange
import time

def bem_setup(a, b, M, phi1, phi2):
    theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
    nodes = np.zeros((M, 2), dtype=np.float64)
    nodes[:, 0] = a * np.cos(theta)
    nodes[:, 1] = b * np.sin(theta)

    elements = np.zeros((M, 2), dtype=np.int32)
    normals = np.zeros((M, 2), dtype=np.float64)
    lengths = np.zeros(M, dtype=np.float64)

    for i in range(M):
        elements[i, 0] = i
        elements[i, 1] = (i + 1) % M

        dx = nodes[(i + 1) % M, 0] - nodes[i, 0]
        dy = nodes[(i + 1) % M, 1] - nodes[i, 1]
        lengths[i] = np.hypot(dx, dy)
        normals[i, 0] = dy / lengths[i]
        normals[i, 1] = -dx / lengths[i]

    num_q = len(phi1)
    q_pts_x = np.zeros((M, num_q), dtype=np.float64)
    q_pts_y = np.zeros((M, num_q), dtype=np.float64)

    for i in range(M):
        nA = elements[i, 0]
        nB = elements[i, 1]
        Ax, Ay = nodes[nA, 0], nodes[nA, 1]
        Bx, By = nodes[nB, 0], nodes[nB, 1]
        for q in range(num_q):
            q_pts_x[i, q] = Ax * phi1[q] + Bx * phi2[q]
            q_pts_y[i, q] = Ay * phi1[q] + By * phi2[q]

    return nodes, elements, normals, lengths, q_pts_x, q_pts_y

@njit(parallel=True, fastmath=True)
def compute_rhs_numba(elements, lengths, q_wts, phi1, phi2, q_pts_x, q_pts_y):
    M = len(elements)
    num_q = len(q_wts)
    F_elem = np.zeros((M, 2), dtype=np.float64)

    for ex in prange(M):
        Jx = lengths[ex] / 2.0
        valA = 0.0
        valB = 0.0

        for qx in range(num_q):
            p1x = phi1[qx]
            p2x = phi2[qx]
            x_val_x = q_pts_x[ex, qx]
            x_val_y = q_pts_y[ex, qx]

            f_val = x_val_x**2 - x_val_y**2
            term = f_val * Jx * q_wts[qx]

            valA += term * p1x
            valB += term * p2x

        F_elem[ex, 0] = valA
        F_elem[ex, 1] = valB

    F = np.zeros(M, dtype=np.float64)
    for ex in range(M):
        nAx = elements[ex, 0]
        nBx = elements[ex, 1]
        F[nAx] += F_elem[ex, 0]
        F[nBx] += F_elem[ex, 1]

    return F

@njit(parallel=True, fastmath=True)
def bem_matvec(sigma, elements, normals, lengths, q_wts, phi1, phi2, q_pts_x, q_pts_y):
    M = len(sigma)
    num_q = len(q_wts)
    out_elem = np.zeros((M, 2), dtype=np.float64)

    out_mass = np.zeros(M, dtype=np.float64)
    for i in prange(M):
        e_prev = (i - 1 + M) % M
        e_next = i
        L_prev = lengths[e_prev]
        L_next = lengths[e_next]
        prev_node = (i - 1 + M) % M
        next_node = (i + 1) % M

        mass_prev = L_prev * (sigma[i] / 3.0 + sigma[prev_node] / 6.0)
        mass_next = L_next * (sigma[i] / 3.0 + sigma[next_node] / 6.0)
        out_mass[i] = 0.5 * (mass_prev + mass_next)

    sig_w_q = np.zeros((M, num_q), dtype=np.float64)
    for ey in prange(M):
        sigA = sigma[elements[ey, 0]]
        sigB = sigma[elements[ey, 1]]
        Jy = lengths[ey] / 2.0
        for qy in range(num_q):
            sig_val = sigA * phi1[qy] + sigB * phi2[qy]
            sig_w_q[ey, qy] = sig_val * Jy * q_wts[qy]

    for ex in prange(M):
        valA = 0.0
        valB = 0.0
        term_qx = lengths[ex] / 2.0

        for qx in range(num_q):
            p1x = phi1[qx]
            p2x = phi2[qx]
            x_val_x = q_pts_x[ex, qx]
            x_val_y = q_pts_y[ex, qx]
            t_qx = term_qx * q_wts[qx]

            sum_K = 0.0
            for ey in range(M):
                if ex == ey:
                    continue

                ny_x = normals[ey, 0]
                ny_y = normals[ey, 1]

                for qy in range(num_q):
                    dx = x_val_x - q_pts_x[ey, qy]
                    dy = x_val_y - q_pts_y[ey, qy]
                    r2 = dx*dx + dy*dy

                    dot = dx * ny_x + dy * ny_y
                    K = -dot / (2.0 * np.pi * r2)

                    sum_K += K * sig_w_q[ey, qy]

            valA += sum_K * t_qx * p1x
            valB += sum_K * t_qx * p2x

        out_elem[ex, 0] = valA
        out_elem[ex, 1] = valB

    out = np.zeros(M, dtype=np.float64)
    for ex in range(M):
        nAx = elements[ex, 0]
        nBx = elements[ex, 1]
        out[nAx] += out_elem[ex, 0]
        out[nBx] += out_elem[ex, 1]

    return out + out_mass

@njit(parallel=True, fastmath=True)
def _eval_chunk_numba(sigma, elements, normals, lengths, q_wts, phi1, phi2, q_pts_x, q_pts_y, eval_X, eval_Y):
    N = len(eval_X)
    M = len(elements)
    num_q = len(q_wts)
    out = np.zeros(N, dtype=np.float64)

    sig_w_q = np.zeros((M, num_q), dtype=np.float64)
    for ey in prange(M):
        sigA = sigma[elements[ey, 0]]
        sigB = sigma[elements[ey, 1]]
        Jy = lengths[ey] / 2.0
        for qy in range(num_q):
            sig_val = sigA * phi1[qy] + sigB * phi2[qy]
            sig_w_q[ey, qy] = sig_val * Jy * q_wts[qy]

    for i in prange(N):
        xi = eval_X[i]
        yi = eval_Y[i]
        val = 0.0

        for ey in range(M):
            nx = normals[ey, 0]
            ny = normals[ey, 1]

            for qy in range(num_q):
                dx = xi - q_pts_x[ey, qy]
                dy = yi - q_pts_y[ey, qy]
                r2 = dx*dx + dy*dy

                dot = dx * nx + dy * ny
                K = -dot / (2.0 * np.pi * r2)

                val += K * sig_w_q[ey, qy]

        out[i] = val

    return out

def bem_evaluate_chunked(sigma, elements, normals, lengths, q_wts, phi1, phi2, q_pts_x, q_pts_y, eval_X, eval_Y, chunk_size=5000):
    N = len(eval_X)
    out = np.zeros(N, dtype=np.float64)
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        out[i:end] = _eval_chunk_numba(sigma, elements, normals, lengths, q_wts, phi1, phi2, q_pts_x, q_pts_y, eval_X[i:end], eval_Y[i:end])
    return out

class GMRESCounter:
    def __init__(self):
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1

def run_bem(M, a, b, eval_X, eval_Y, u_exact, q_wts, phi1, phi2):
    t0 = time.perf_counter()
    nodes, elements, normals, lengths, q_pts_x, q_pts_y = bem_setup(a, b, M, phi1, phi2)
    F_rhs = compute_rhs_numba(elements, lengths, q_wts, phi1, phi2, q_pts_x, q_pts_y)
    t_setup = time.perf_counter() - t0

    t1 = time.perf_counter()
    counter = GMRESCounter()

    def matvec_wrapper(sigma_vec):
        return bem_matvec(sigma_vec, elements, normals, lengths, q_wts, phi1, phi2, q_pts_x, q_pts_y)

    A_op = spla.LinearOperator((M, M), matvec=matvec_wrapper, dtype=np.float64)

    sigma_sol, _ = spla.gmres(A_op, F_rhs, rtol=1e-8, callback=counter, callback_type='pr_norm')
    t_solve = time.perf_counter() - t1

    t2 = time.perf_counter()
    u_approx = bem_evaluate_chunked(sigma_sol, elements, normals, lengths, q_wts, phi1, phi2, q_pts_x, q_pts_y, eval_X, eval_Y)
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

    q_pts, q_wts = np.polynomial.legendre.leggauss(4)
    phi1 = 0.5 * (1.0 - q_pts)
    phi2 = 0.5 * (1.0 + q_pts)

    n_w, e_w, norm_w, len_w, qx_w, qy_w = bem_setup(a, b, 10, phi1, phi2)
    compute_rhs_numba(e_w, len_w, q_wts, phi1, phi2, qx_w, qy_w)
    bem_matvec(np.zeros(10), e_w, norm_w, len_w, q_wts, phi1, phi2, qx_w, qy_w)
    _eval_chunk_numba(np.zeros(10), e_w, norm_w, len_w, q_wts, phi1, phi2, qx_w, qy_w, eval_X[:2], eval_Y[:2])

    M_values = [4000, 8000, 16000, 32000]

    print(f"{'M':>10} | {'Iters':>8} | {'Setup(s)':>10} | {'Solve(s)':>10} | {'Eval(s)':>10} | {'Total(s)':>10} | {'Rel L2 Error':>15}")
    print("-" * 87)

    for M in M_values:
        res = run_bem(M, a, b, eval_X, eval_Y, u_exact, q_wts, phi1, phi2)
        print(f"{res['M']:10d} | {res['iterations']:8d} | {res['setup_time']:10.4f} | {res['solve_time']:10.4f} | {res['eval_time']:10.4f} | {res['total_time']:10.4f} | {res['relative_L2_error']:15.4e}")

if __name__ == "__main__":
    main()