import numpy as np
import time
from scipy.sparse.linalg import LinearOperator, gmres
from numba import njit

# --- Precomputed Gauss Quadrature (Order 10) ---
gauss_pts = np.array([
    -0.9739065285171717, -0.8650633666889845, -0.6794095682990244,
    -0.4333953941292472, -0.1488743389816312,  0.1488743389816312,
     0.4333953941292472,  0.6794095682990244,  0.8650633666889845,
     0.9739065285171717
])
gauss_wts = np.array([
     0.0666713443086881,  0.1494513491505806,  0.2190863625159821,
     0.2692667193099963,  0.2955242247147529,  0.2955242247147529,
     0.2692667193099963,  0.2190863625159821,  0.1494513491505806,
     0.0666713443086881
])

def build_geometry(N):
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    Apts = []
    Bpts = []

    # Cosine spacing for each edge
    for i in range(3):
        V_start = vertices[i]
        V_end = vertices[(i+1)%3]

        t_vals = (1.0 - np.cos(np.arange(N + 1) * np.pi / N)) / 2.0

        for j in range(N):
            A = V_start + t_vals[j] * (V_end - V_start)
            B = V_start + t_vals[j+1] * (V_end - V_start)
            Apts.append(A)
            Bpts.append(B)

    Apts = np.array(Apts)
    Bpts = np.array(Bpts)

    num_panels = 3 * N
    centers = (Apts + Bpts) / 2.0
    normals = np.zeros((num_panels, 2))
    lengths = np.zeros(num_panels)

    for e in range(num_panels):
        dx = Bpts[e, 0] - Apts[e, 0]
        dy = Bpts[e, 1] - Apts[e, 1]
        L = np.sqrt(dx**2 + dy**2)
        lengths[e] = L
        # Inward normal for CCW traversal
        normals[e, 0] = -dy / L
        normals[e, 1] = dx / L

    return centers, normals, lengths, Apts, Bpts

def precompute(Apts, Bpts, lengths):
    num_panels = len(Apts)
    Nq = len(gauss_pts)

    y_q = np.zeros((num_panels, Nq, 2))
    ds_q = np.zeros(num_panels)

    phi1 = (1.0 - gauss_pts) / 2.0
    phi2 = (1.0 + gauss_pts) / 2.0

    for e in range(num_panels):
        ds_q[e] = lengths[e] / 2.0  # L/2 from Jacobian

        for k in range(Nq):
            y_q[e, k, 0] = phi1[k] * Apts[e, 0] + phi2[k] * Bpts[e, 0]
            y_q[e, k, 1] = phi1[k] * Apts[e, 1] + phi2[k] * Bpts[e, 1]

    return y_q, ds_q

@njit
def matvec_njit(mu, centers, normals, lengths, y_q, ds_q, alpha, gauss_wts):
    num_panels = len(centers)
    Nq = len(gauss_wts)
    y = np.zeros(num_panels)

    for i in range(num_panels):
        xi_x = centers[i, 0]
        xi_y = centers[i, 1]

        # Jump term (1/2 I)
        y[i] = 0.5 * mu[i]

        sum_i = 0.0
        for e in range(num_panels):
            if i == e:
                # Analytical self-interaction for flat panels
                K_self = 0.0
                V_self = -(lengths[e] / (2.0 * np.pi)) * (np.log(lengths[e] / 2.0) - 1.0)
                sum_i += (K_self + alpha * V_self) * mu[e]
            else:
                nx = normals[e, 0]
                ny = normals[e, 1]

                val_e = 0.0
                for k in range(Nq):
                    dx = xi_x - y_q[e, k, 0]
                    dy = xi_y - y_q[e, k, 1]

                    dist2 = dx**2 + dy**2

                    # Double layer dG/dn
                    dot = dx * nx + dy * ny
                    K_val = (1.0 / (2.0 * np.pi)) * (dot / dist2)

                    # Single layer G
                    V_val = -(1.0 / (4.0 * np.pi)) * np.log(dist2)

                    val_e += (K_val + alpha * V_val) * gauss_wts[k]

                sum_i += val_e * ds_q[e] * mu[e]

        y[i] += sum_i

    return y

@njit
def evaluate_interior(pts, mu, centers, normals, lengths, y_q, ds_q, alpha, gauss_wts):
    num_pts = len(pts)
    num_panels = len(centers)
    Nq = len(gauss_wts)

    u = np.zeros(num_pts)

    for i in range(num_pts):
        xi_x = pts[i, 0]
        xi_y = pts[i, 1]

        sum_i = 0.0
        for e in range(num_panels):
            nx = normals[e, 0]
            ny = normals[e, 1]

            val_e = 0.0
            for k in range(Nq):
                dx = xi_x - y_q[e, k, 0]
                dy = xi_y - y_q[e, k, 1]

                dist2 = dx**2 + dy**2

                dot = dx * nx + dy * ny
                K_val = (1.0 / (2.0 * np.pi)) * (dot / dist2)

                V_val = -(1.0 / (4.0 * np.pi)) * np.log(dist2)

                val_e += (K_val + alpha * V_val) * gauss_wts[k]

            sum_i += val_e * ds_q[e] * mu[e]

        u[i] = sum_i

    return u

class GMRESCounter:
    def __init__(self):
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1

def main():
    alpha = 0.1
    ngrid = 60
    x_1d = np.linspace(0.02, 0.98, ngrid)
    y_1d = np.linspace(0.02, 0.98, ngrid)
    X, Y = np.meshgrid(x_1d, y_1d)
    X_flat = X.flatten()
    Y_flat = Y.flatten()

    # Evaluate only inside triangle
    mask = (X_flat + Y_flat) < 1.0
    eval_pts = np.vstack((X_flat[mask], Y_flat[mask])).T

    u_exact = np.sin(np.pi * eval_pts[:, 0]) * np.cosh(np.pi * eval_pts[:, 1])

    # Numba compilation warm-up
    c_w, n_w, l_w, A_w, B_w = build_geometry(4)
    y_q_w, ds_q_w = precompute(A_w, B_w, l_w)
    mu_w = np.zeros(len(c_w))
    _ = matvec_njit(mu_w, c_w, n_w, l_w, y_q_w, ds_q_w, alpha, gauss_wts)
    _ = evaluate_interior(eval_pts[:2], mu_w, c_w, n_w, l_w, y_q_w, ds_q_w, alpha, gauss_wts)

    print(f"{'N':<5} | {'Unknowns':<8} | {'Iters':<6} | {'L2 Rel Error':<13} | {'Setup(s)':<9} | {'Solve(s)':<9} | {'Eval(s)':<9} | {'Total(s)'}")
    print("-" * 88)

    for N in [25, 50, 100, 200, 400]:
        t_start = time.time()

        # Setup
        t0 = time.time()
        centers, normals, lengths, Apts, Bpts = build_geometry(N)
        y_q, ds_q = precompute(Apts, Bpts, lengths)
        num_panels = len(centers)
        t_setup = time.time() - t0

        # Solve
        t0 = time.time()
        def mv(v):
            return matvec_njit(v, centers, normals, lengths, y_q, ds_q, alpha, gauss_wts)

        A_op = LinearOperator((num_panels, num_panels), matvec=mv, dtype=np.float64)
        f = np.sin(np.pi * centers[:, 0]) * np.cosh(np.pi * centers[:, 1])

        counter = GMRESCounter()
        mu, info = gmres(A_op, f, callback=counter, atol=1e-10, callback_type= 'legacy')
        t_solve = time.time() - t0

        # Evaluate interior
        t0 = time.time()
        u_approx = evaluate_interior(eval_pts, mu, centers, normals, lengths, y_q, ds_q, alpha, gauss_wts)
        t_eval = time.time() - t0

        t_total = time.time() - t_start

        # Error metrics
        err = np.linalg.norm(u_approx - u_exact) / np.linalg.norm(u_exact)

        print(f"{N:<5} | {num_panels:<8} | {counter.niter:<6} | {err:<13.6e} | {t_setup:<9.4f} | {t_solve:<9.4f} | {t_eval:<9.4f} | {t_total:<9.4f}")

if __name__ == '__main__':
    main()
