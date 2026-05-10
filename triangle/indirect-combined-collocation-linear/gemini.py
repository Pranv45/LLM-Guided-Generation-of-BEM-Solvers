import numpy as np
import time
from scipy.sparse.linalg import LinearOperator, gmres
from numba import njit

# --- Precomputed Gauss Quadrature (Order 8) ---
gauss_pts = np.array([
    -0.960289856497536, -0.796666477413627, -0.525532409916329, -0.183434642495650,
     0.183434642495650,  0.525532409916329,  0.796666477413627,  0.960289856497536
])
gauss_wts = np.array([
     0.101228536290376,  0.222381034453374,  0.313706645877887,  0.362683783378362,
     0.362683783378362,  0.313706645877887,  0.222381034453374,  0.101228536290376
])

phi1 = (1.0 - gauss_pts) / 2.0
phi2 = (1.0 + gauss_pts) / 2.0

def build_geometry(N):
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    nodes = []

    # Cosine spacing for each edge
    for i in range(3):
        A = vertices[i]
        B = vertices[(i+1)%3]
        for j in range(N):
            t = (1.0 - np.cos(j * np.pi / N)) / 2.0
            nodes.append(A + t * (B - A))

    nodes = np.array(nodes)
    num_nodes = 3 * N

    elements = np.zeros((num_nodes, 2), dtype=np.int32)
    for e in range(num_nodes):
        elements[e, 0] = e
        elements[e, 1] = (e + 1) % num_nodes

    return nodes, elements

def precompute(nodes, elements):
    num_elements = len(elements)
    Nq = len(gauss_pts)

    normals = np.zeros((num_elements, 2))
    lengths = np.zeros(num_elements)
    y_q = np.zeros((num_elements, Nq, 2))
    ds_q = np.zeros(num_elements)

    for e in range(num_elements):
        A = nodes[elements[e, 0]]
        B = nodes[elements[e, 1]]
        dx = B[0] - A[0]
        dy = B[1] - A[1]
        L = np.sqrt(dx**2 + dy**2)
        lengths[e] = L

        # Inward normal for CCW traversal
        normals[e, 0] = -dy / L
        normals[e, 1] = dx / L

        # L/2 from Jacobian, 1/(2pi) from Fundamental Solution
        ds_q[e] = L / (4.0 * np.pi)

        for k in range(Nq):
            y_q[e, k, 0] = phi1[k] * A[0] + phi2[k] * B[0]
            y_q[e, k, 1] = phi1[k] * A[1] + phi2[k] * B[1]

    return normals, lengths, y_q, ds_q

@njit
def matvec_njit(mu, nodes, elements, y_q, normals, ds_q, phi1, phi2, gauss_wts):
    num_nodes = len(nodes)
    num_elements = len(elements)
    Nq = len(gauss_wts)

    y = np.zeros(num_nodes)

    for i in range(num_nodes):
        xi_x = nodes[i, 0]
        xi_y = nodes[i, 1]

        # 1/2 I term
        y[i] = 0.5 * mu[i]

        sum_i = 0.0
        for e in range(num_elements):
            muA = mu[elements[e, 0]]
            muB = mu[elements[e, 1]]
            nx = normals[e, 0]
            ny = normals[e, 1]
            ds = ds_q[e]

            val_A = 0.0
            val_B = 0.0
            for k in range(Nq):
                dx = xi_x - y_q[e, k, 0]
                dy = xi_y - y_q[e, k, 1]

                dist2 = dx**2 + dy**2

                if dist2 > 1e-14:
                    dot = dx * nx + dy * ny
                    kernel = dot / dist2
                    term = ds * kernel * gauss_wts[k]

                    val_A += term * phi1[k]
                    val_B += term * phi2[k]

            sum_i += val_A * muA + val_B * muB

        y[i] += sum_i

    return y

@njit
def evaluate_interior(pts, mu, elements, y_q, normals, ds_q, phi1, phi2, gauss_wts):
    num_pts = len(pts)
    num_elements = len(elements)
    Nq = len(gauss_wts)

    u = np.zeros(num_pts)

    for i in range(num_pts):
        xi_x = pts[i, 0]
        xi_y = pts[i, 1]

        sum_i = 0.0
        for e in range(num_elements):
            muA = mu[elements[e, 0]]
            muB = mu[elements[e, 1]]
            nx = normals[e, 0]
            ny = normals[e, 1]
            ds = ds_q[e]

            val_A = 0.0
            val_B = 0.0
            for k in range(Nq):
                dx = xi_x - y_q[e, k, 0]
                dy = xi_y - y_q[e, k, 1]

                dist2 = dx**2 + dy**2

                dot = dx * nx + dy * ny
                kernel = dot / dist2
                term = ds * kernel * gauss_wts[k]

                val_A += term * phi1[k]
                val_B += term * phi2[k]

            sum_i += val_A * muA + val_B * muB

        u[i] = sum_i

    return u

class GMRESCounter:
    def __init__(self):
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1

def main():
    ngrid = 60
    x_1d = np.linspace(0.02, 0.98, ngrid)
    y_1d = np.linspace(0.02, 0.98, ngrid)
    X, Y = np.meshgrid(x_1d, y_1d)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    mask = (X_flat + Y_flat) < 1.0
    eval_pts = np.vstack((X_flat[mask], Y_flat[mask])).T

    u_exact = np.sin(np.pi * eval_pts[:, 0]) * np.cosh(np.pi * eval_pts[:, 1])

    # Numba compilation warm-up
    n_w, e_w = build_geometry(4)
    norm_w, len_w, y_q_w, ds_q_w = precompute(n_w, e_w)
    mu_w = np.zeros(len(n_w))
    _ = matvec_njit(mu_w, n_w, e_w, y_q_w, norm_w, ds_q_w, phi1, phi2, gauss_wts)
    _ = evaluate_interior(eval_pts[:2], mu_w, e_w, y_q_w, norm_w, ds_q_w, phi1, phi2, gauss_wts)

    print(f"{'N':<5} | {'Unknowns':<8} | {'Iters':<6} | {'L2 Rel Error':<13} | {'Setup(s)':<9} | {'Solve(s)':<9} | {'Eval(s)':<9} | {'Total(s)'}")
    print("-" * 88)

    for N in [25, 50, 100, 200, 400]:
        t_start = time.time()

        # Setup
        t0 = time.time()
        nodes, elements = build_geometry(N)
        normals, lengths, y_q, ds_q = precompute(nodes, elements)
        num_nodes = len(nodes)
        t_setup = time.time() - t0

        # Solve
        t0 = time.time()
        def mv(v):
            return matvec_njit(v, nodes, elements, y_q, normals, ds_q, phi1, phi2, gauss_wts)

        A_op = LinearOperator((num_nodes, num_nodes), matvec=mv, dtype=np.float64)
        f = np.sin(np.pi * nodes[:, 0]) * np.cosh(np.pi * nodes[:, 1])

        counter = GMRESCounter()
        mu, info = gmres(A_op, f, callback=counter, atol=1e-10, callback_type= 'legacy')
        t_solve = time.time() - t0

        # Evaluate interior
        t0 = time.time()
        u_approx = evaluate_interior(eval_pts, mu, elements, y_q, normals, ds_q, phi1, phi2, gauss_wts)
        t_eval = time.time() - t0

        t_total = time.time() - t_start

        # Error metrics
        err = np.linalg.norm(u_approx - u_exact) / np.linalg.norm(u_exact)

        print(f"{N:<5} | {num_nodes:<8} | {counter.niter:<6} | {err:<13.6e} | {t_setup:<9.4f} | {t_solve:<9.4f} | {t_eval:<9.4f} | {t_total:<9.4f}")

if __name__ == '__main__':
    main()