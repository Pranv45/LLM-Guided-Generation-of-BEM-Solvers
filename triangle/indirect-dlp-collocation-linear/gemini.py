import numpy as np
import scipy.sparse.linalg as spla
from numba import njit
import time

@njit
def exact_solution(x, y):
    return np.sin(np.pi * x) * np.cosh(np.pi * y)

@njit
def matvec_numba(sigma, nodes, elements, normals, lengths, sq, wq, eta):
    M = nodes.shape[0]
    num_elems = elements.shape[0]
    num_q = sq.shape[0]
    out = np.zeros(M)

    phi1 = 0.5 * (1.0 - sq)
    phi2 = 0.5 * (1.0 + sq)

    for i in range(M):
        xi = nodes[i, 0]
        yi = nodes[i, 1]
        val = 0.0

        for j in range(num_elems):
            nA = elements[j, 0]
            nB = elements[j, 1]
            Ax, Ay = nodes[nA, 0], nodes[nA, 1]
            Bx, By = nodes[nB, 0], nodes[nB, 1]
            nx, ny = normals[j, 0], normals[j, 1]
            L = lengths[j]

            sigA = sigma[nA]
            sigB = sigma[nB]

            J = L / 2.0

            for q in range(num_q):
                yqx = Ax * phi1[q] + Bx * phi2[q]
                yqy = Ay * phi1[q] + By * phi2[q]

                dx = xi - yqx
                dy = yi - yqy
                r2 = dx*dx + dy*dy
                r = np.sqrt(r2)

                if r > 1e-14:
                    G = - (1.0 / (2.0 * np.pi)) * np.log(r)
                    dGdn = (1.0 / (2.0 * np.pi)) * (dx * nx + dy * ny) / r2

                    kernel = G + eta * dGdn
                    sig_val = sigA * phi1[q] + sigB * phi2[q]
                    val += kernel * sig_val * wq[q] * J

        out[i] = val - (eta / 2.0) * sigma[i]

    return out

@njit
def eval_interior_numba(sigma, nodes, elements, normals, lengths, sq, wq, eta, X, Y):
    K = X.shape[0]
    num_elems = elements.shape[0]
    num_q = sq.shape[0]
    out = np.zeros(K)

    phi1 = 0.5 * (1.0 - sq)
    phi2 = 0.5 * (1.0 + sq)

    for k in range(K):
        xi = X[k]
        yi = Y[k]
        val = 0.0

        for j in range(num_elems):
            nA = elements[j, 0]
            nB = elements[j, 1]
            Ax, Ay = nodes[nA, 0], nodes[nA, 1]
            Bx, By = nodes[nB, 0], nodes[nB, 1]
            nx, ny = normals[j, 0], normals[j, 1]
            L = lengths[j]

            sigA = sigma[nA]
            sigB = sigma[nB]

            J = L / 2.0

            for q in range(num_q):
                yqx = Ax * phi1[q] + Bx * phi2[q]
                yqy = Ay * phi1[q] + By * phi2[q]

                dx = xi - yqx
                dy = yi - yqy
                r2 = dx*dx + dy*dy
                r = np.sqrt(r2)

                if r > 1e-14:
                    G = - (1.0 / (2.0 * np.pi)) * np.log(r)
                    dGdn = (1.0 / (2.0 * np.pi)) * (dx * nx + dy * ny) / r2

                    kernel = G + eta * dGdn
                    sig_val = sigA * phi1[q] + sigB * phi2[q]
                    val += kernel * sig_val * wq[q] * J

        out[k] = val

    return out

class GMRESCounter:
    def __init__(self):
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1

def build_geometry(N):
    nodes_list = []
    for i in range(N):
        t = 0.5 * (1.0 - np.cos(i * np.pi / N))
        nodes_list.append([t, 0.0])
    for i in range(N):
        t = 0.5 * (1.0 - np.cos(i * np.pi / N))
        nodes_list.append([1.0 - t, t])
    for i in range(N):
        t = 0.5 * (1.0 - np.cos(i * np.pi / N))
        nodes_list.append([0.0, 1.0 - t])

    nodes = np.array(nodes_list, dtype=np.float64)
    M = nodes.shape[0]

    elements = np.zeros((M, 2), dtype=np.int64)
    for i in range(M):
        elements[i, 0] = i
        elements[i, 1] = (i + 1) % M

    lengths = np.zeros(M, dtype=np.float64)
    normals = np.zeros((M, 2), dtype=np.float64)
    for i in range(M):
        nA, nB = elements[i, 0], elements[i, 1]
        dx = nodes[nB, 0] - nodes[nA, 0]
        dy = nodes[nB, 1] - nodes[nA, 1]
        L = np.sqrt(dx**2 + dy**2)
        lengths[i] = L
        normals[i, 0] = dy / L
        normals[i, 1] = -dx / L

    return nodes, elements, normals, lengths

def main():
    sq, wq = np.polynomial.legendre.leggauss(10)
    sq = sq.astype(np.float64)
    wq = wq.astype(np.float64)
    eta = 1.0

    # Proper dummy arrays to pre-compile numba functions
    dummy_nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    dummy_elems = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
    dummy_normals = np.array([[0.0, -1.0], [0.707, 0.707], [-1.0, 0.0]], dtype=np.float64)
    dummy_lengths = np.array([1.0, 1.414, 1.0], dtype=np.float64)
    dummy_vec = np.zeros(3, dtype=np.float64)
    dummy_X = np.array([0.5], dtype=np.float64)

    matvec_numba(dummy_vec, dummy_nodes, dummy_elems, dummy_normals, dummy_lengths, sq, wq, eta)
    eval_interior_numba(dummy_vec, dummy_nodes, dummy_elems, dummy_normals, dummy_lengths, sq, wq, eta, dummy_X, dummy_X)

    ngrid = 60
    x1d = np.linspace(0.02, 0.98, ngrid)
    y1d = np.linspace(0.02, 0.98, ngrid)
    XX, YY = np.meshgrid(x1d, y1d)
    X_flat = XX.ravel()
    Y_flat = YY.ravel()
    mask = (X_flat + Y_flat) < 1.0
    X_int = X_flat[mask].astype(np.float64)
    Y_int = Y_flat[mask].astype(np.float64)
    u_exact_int = exact_solution(X_int, Y_int)

    N_values = [100, 200, 400, 800]

    print(f"{'Unknowns':>10} | {'GMRES It':>10} | {'L2 Error':>12} | {'Setup(s)':>10} | {'Solve(s)':>10} | {'Eval(s)':>10} | {'Total(s)':>10}")
    print("-" * 95)

    for N in N_values:
        t0 = time.perf_counter()

        nodes, elements, normals, lengths = build_geometry(N)
        M = nodes.shape[0]

        f = exact_solution(nodes[:, 0], nodes[:, 1])

        def matvec_wrapper(sigma):
            return matvec_numba(sigma, nodes, elements, normals, lengths, sq, wq, eta)

        A_op = spla.LinearOperator((M, M), matvec=matvec_wrapper)

        t_setup = time.perf_counter() - t0

        t1 = time.perf_counter()
        counter = GMRESCounter()
        sigma_sol, info = spla.gmres(A_op, f, rtol=1e-8, callback=counter, callback_type='legacy')
        t_solve = time.perf_counter() - t1

        t2 = time.perf_counter()
        u_approx_int = eval_interior_numba(sigma_sol, nodes, elements, normals, lengths, sq, wq, eta, X_int, Y_int)
        t_eval = time.perf_counter() - t2

        t_total = t_setup + t_solve + t_eval

        l2_rel_err = np.linalg.norm(u_approx_int - u_exact_int) / np.linalg.norm(u_exact_int)

        print(f"{M:10d} | {counter.niter:10d} | {l2_rel_err:12.4e} | {t_setup:10.4f} | {t_solve:10.4f} | {t_eval:10.4f} | {t_total:10.4f}")

if __name__ == "__main__":
    main()