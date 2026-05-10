#no numba acceleration
import numpy as np
import time
from scipy.sparse.linalg import LinearOperator, gmres
from numpy.polynomial.legendre import leggauss

# ============================================================
# BEM Setup
# ============================================================

def bem_setup(M, n_mode, quad_order=8):
    t0 = time.perf_counter()

    theta = np.linspace(0, 2*np.pi, M, endpoint=False)
    nodes = np.column_stack((np.cos(theta), np.sin(theta)))
    normals = nodes.copy()

    elem_nodes = np.column_stack((np.arange(M), (np.arange(M)+1)%M))
    x0 = nodes[elem_nodes[:,0]]
    x1 = nodes[elem_nodes[:,1]]
    elem_vec = x1 - x0
    elem_len = np.linalg.norm(elem_vec, axis=1)

    s, w = leggauss(quad_order)
    phi1 = (1 - s)/2
    phi2 = (1 + s)/2

    quad_pts = x0[:,None,:] + ((s+1)/2)[None,:,None] * elem_vec[:,None,:]
    quad_w = w[None,:] * (elem_len/2)[:,None]

    f = np.cos(n_mode * theta)

    setup_time = time.perf_counter() - t0

    return {
        "nodes": nodes,
        "normals": normals,
        "elem_nodes": elem_nodes,
        "quad_pts": quad_pts,
        "quad_w": quad_w,
        "phi1": phi1,
        "phi2": phi2,
        "f": f,
        "setup_time": setup_time
    }

# ============================================================
# Matrix-free matvec
# ============================================================

def bem_matvec(mu, data):
    nodes = data["nodes"]
    normals = data["normals"]
    elem_nodes = data["elem_nodes"]
    quad_pts = data["quad_pts"]
    quad_w = data["quad_w"]
    phi1 = data["phi1"]
    phi2 = data["phi2"]

    M = len(nodes)
    result = 0.5 * mu.copy()

    for e in range(M):
        n0, n1 = elem_nodes[e]
        mu_q = phi1 * mu[n0] + phi2 * mu[n1]

        diff = nodes[:,None,:] - quad_pts[e][None,:,:]
        r2 = np.sum(diff**2, axis=2)
        mask = r2 > 1e-14

        kernel = np.zeros_like(r2)
        dot = np.sum(diff * normals[n0], axis=2)
        kernel[mask] = -(1/(2*np.pi)) * dot[mask] / r2[mask]

        contrib = np.sum(kernel * mu_q[None,:] * quad_w[e][None,:], axis=1)
        result += contrib

    return result

# ============================================================
# Interior evaluation (Cartesian masked grid)
# ============================================================

def bem_evaluate_chunked(mu, data, n_mode, chunk=2000):
    t0 = time.perf_counter()

    # ---- Cartesian evaluation grid (your version) ----
    Nx = Ny = 120
    xx = np.linspace(-0.9, 0.9, Nx)
    yy = np.linspace(-0.9, 0.9, Ny)
    X, Y = np.meshgrid(xx, yy)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    mask = (X**2 + Y**2) < 0.9**2
    eval_pts = pts[mask.ravel()]

    # ---------------------------------------------------

    elem_nodes = data["elem_nodes"]
    quad_pts = data["quad_pts"]
    quad_w = data["quad_w"]
    phi1 = data["phi1"]
    phi2 = data["phi2"]

    u_num = np.zeros(len(eval_pts))

    # Interpolate μ to quadrature points
    mu_elem = phi1[None,:]*mu[elem_nodes[:,0],None] + \
              phi2[None,:]*mu[elem_nodes[:,1],None]

    quad_pts_all = quad_pts.reshape(-1,2)
    quad_w_all = quad_w.reshape(-1)
    mu_all = mu_elem.reshape(-1)

    # ---- Chunked vectorized evaluation ----
    for i in range(0, len(eval_pts), chunk):
        xe = eval_pts[i:i+chunk]

        diff = xe[:,None,:] - quad_pts_all[None,:,:]
        r2 = np.sum(diff**2, axis=2)
        dot = np.sum(diff * quad_pts_all[None,:,:], axis=2)

        kernel = -(1/(2*np.pi)) * dot / r2

        u_num[i:i+chunk] = np.sum(
            kernel * mu_all[None,:] * quad_w_all[None,:],
            axis=1
        )

    # ---- Exact solution in Cartesian coordinates ----
    x = eval_pts[:,0]
    y = eval_pts[:,1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    u_exact = (r**n_mode) * np.cos(n_mode*theta)

    rel_L2 = np.linalg.norm(u_num - u_exact) / np.linalg.norm(u_exact)
    Linf = np.max(np.abs(u_num - u_exact))

    eval_time = time.perf_counter() - t0

    return rel_L2, Linf, eval_time

# ============================================================
# Run BEM
# ============================================================

def run_bem(M, n_mode):
    data = bem_setup(M, n_mode)
    f = data["f"]

    t0 = time.perf_counter()

    iter_count = [0]
    def callback(res):
        iter_count[0] += 1

    A = LinearOperator((M,M), matvec=lambda x: bem_matvec(x, data))
    mu, _ = gmres(A, f, atol=1e-8, callback=callback)

    solve_time = time.perf_counter() - t0

    rel_L2, Linf, eval_time = bem_evaluate_chunked(mu, data, n_mode)

    total_time = data["setup_time"] + solve_time + eval_time

    return {
        "M": M,
        "iterations": iter_count[0],
        "setup_time": data["setup_time"],
        "solve_time": solve_time,
        "eval_time": eval_time,
        "total_time": total_time,
        "relative_L2_error": rel_L2,
        "Linf_error": Linf
    }

# ============================================================
# Main Sweep
# ============================================================
print("no numba acceleration")
if __name__ == "__main__":
    n_mode = 3
    Ms = [1000, 2000, 4000, 8000, 16000]

    print(" M      Iter    Setup(s)  Solve(s)  Eval(s)   Total(s)   RelL2       Linf")
    print("-"*80)

    for M in Ms:
        res = run_bem(M, n_mode)
        print(f"{res['M']:6d}  {res['iterations']:6d}  "
              f"{res['setup_time']:8.3f}  {res['solve_time']:8.3f}  "
              f"{res['eval_time']:8.3f}  {res['total_time']:9.3f}  "
              f"{res['relative_L2_error']:9.2e}  {res['Linf_error']:9.2e}")



# numba acceleration
import numpy as np
import time
from scipy.sparse.linalg import LinearOperator, gmres
from numpy.polynomial.legendre import leggauss
from numba import njit, prange

# ============================================================
# Numba-accelerated kernels
# ============================================================

@njit(cache=True, fastmath=True, parallel=True)
def bem_matvec_numba(mu, nodes, normals, elem_nodes, quad_pts, quad_w, phi1, phi2):
    M = nodes.shape[0]
    Q = phi1.shape[0]
    result = np.empty(M, dtype=np.float64)

    c = -(1.0 / (2.0 * np.pi))

    for i in prange(M):
        acc = 0.5 * mu[i]

        for e in range(M):
            n0 = elem_nodes[e, 0]
            n1 = elem_nodes[e, 1]

            nx = normals[n0, 0]
            ny = normals[n0, 1]

            for q in range(Q):
                mu_q = phi1[q] * mu[n0] + phi2[q] * mu[n1]

                dx = nodes[i, 0] - quad_pts[e, q, 0]
                dy = nodes[i, 1] - quad_pts[e, q, 1]
                r2 = dx * dx + dy * dy

                if r2 > 1e-14:
                    dot = dx * nx + dy * ny
                    acc += c * dot / r2 * mu_q * quad_w[e, q]

        result[i] = acc

    return result


@njit(cache=True, fastmath=True, parallel=True)
def bem_eval_chunk_numba(eval_pts, mu_all, quad_pts_all, quad_w_all):
    n_eval = eval_pts.shape[0]
    n_src = quad_pts_all.shape[0]
    u_num = np.empty(n_eval, dtype=np.float64)

    c = -(1.0 / (2.0 * np.pi))

    for i in prange(n_eval):
        xi = eval_pts[i, 0]
        yi = eval_pts[i, 1]
        acc = 0.0

        for j in range(n_src):
            dx = xi - quad_pts_all[j, 0]
            dy = yi - quad_pts_all[j, 1]
            r2 = dx * dx + dy * dy

            # Kept exactly in the same spirit as your current code:
            # kernel = -(1/(2*pi)) * dot / r2
            dot = dx * quad_pts_all[j, 0] + dy * quad_pts_all[j, 1]
            kernel = c * dot / r2
            acc += kernel * mu_all[j] * quad_w_all[j]

        u_num[i] = acc

    return u_num

# ============================================================
# BEM Setup
# ============================================================

def bem_setup(M, n_mode, quad_order=8):
    t0 = time.perf_counter()

    theta = np.linspace(0, 2*np.pi, M, endpoint=False)
    nodes = np.column_stack((np.cos(theta), np.sin(theta)))
    normals = nodes.copy()

    elem_nodes = np.column_stack((np.arange(M), (np.arange(M)+1)%M))
    x0 = nodes[elem_nodes[:,0]]
    x1 = nodes[elem_nodes[:,1]]
    elem_vec = x1 - x0
    elem_len = np.linalg.norm(elem_vec, axis=1)

    s, w = leggauss(quad_order)
    phi1 = (1 - s)/2
    phi2 = (1 + s)/2

    quad_pts = x0[:,None,:] + ((s+1)/2)[None,:,None] * elem_vec[:,None,:]
    quad_w = w[None,:] * (elem_len/2)[:,None]

    f = np.cos(n_mode * theta)

    setup_time = time.perf_counter() - t0

    return {
        "nodes": nodes,
        "normals": normals,
        "elem_nodes": elem_nodes,
        "quad_pts": quad_pts,
        "quad_w": quad_w,
        "phi1": phi1,
        "phi2": phi2,
        "f": f,
        "setup_time": setup_time
    }

# ============================================================
# Matrix-free matvec
# ============================================================

def bem_matvec(mu, data):
    return bem_matvec_numba(
        mu,
        data["nodes"],
        data["normals"],
        data["elem_nodes"],
        data["quad_pts"],
        data["quad_w"],
        data["phi1"],
        data["phi2"]
    )

# ============================================================
# Interior evaluation (Cartesian masked grid)
# ============================================================

def bem_evaluate_chunked(mu, data, n_mode, chunk=2000):
    t0 = time.perf_counter()

    # ---- Cartesian evaluation grid (your version) ----
    Nx = Ny = 120
    xx = np.linspace(-0.9, 0.9, Nx)
    yy = np.linspace(-0.9, 0.9, Ny)
    X, Y = np.meshgrid(xx, yy)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    mask = (X**2 + Y**2) < 0.9**2
    eval_pts = pts[mask.ravel()]
    # ---------------------------------------------------

    elem_nodes = data["elem_nodes"]
    quad_pts = data["quad_pts"]
    quad_w = data["quad_w"]
    phi1 = data["phi1"]
    phi2 = data["phi2"]

    # Interpolate μ to quadrature points (same as your code)
    mu_elem = phi1[None,:]*mu[elem_nodes[:,0],None] + \
              phi2[None,:]*mu[elem_nodes[:,1],None]

    quad_pts_all = quad_pts.reshape(-1,2)
    quad_w_all = quad_w.reshape(-1)
    mu_all = mu_elem.reshape(-1)

    u_num = np.zeros(len(eval_pts))

    # Chunking retained; each chunk is computed by Numba
    for i in range(0, len(eval_pts), chunk):
        xe = eval_pts[i:i+chunk]
        u_num[i:i+chunk] = bem_eval_chunk_numba(xe, mu_all, quad_pts_all, quad_w_all)

    # ---- Exact solution in Cartesian coordinates ----
    x = eval_pts[:,0]
    y = eval_pts[:,1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    u_exact = (r**n_mode) * np.cos(n_mode*theta)

    rel_L2 = np.linalg.norm(u_num - u_exact) / np.linalg.norm(u_exact)
    Linf = np.max(np.abs(u_num - u_exact))

    eval_time = time.perf_counter() - t0

    return rel_L2, Linf, eval_time

# ============================================================
# Run BEM
# ============================================================

def run_bem(M, n_mode):
    data = bem_setup(M, n_mode)
    f = data["f"]

    t0 = time.perf_counter()

    iter_count = [0]
    def callback(res):
        iter_count[0] += 1

    A = LinearOperator((M,M), matvec=lambda x: bem_matvec(x, data))
    mu, _ = gmres(A, f, atol=1e-8, callback=callback)

    solve_time = time.perf_counter() - t0

    rel_L2, Linf, eval_time = bem_evaluate_chunked(mu, data, n_mode)

    total_time = data["setup_time"] + solve_time + eval_time

    return {
        "M": M,
        "iterations": iter_count[0],
        "setup_time": data["setup_time"],
        "solve_time": solve_time,
        "eval_time": eval_time,
        "total_time": total_time,
        "relative_L2_error": rel_L2,
        "Linf_error": Linf
    }

# ============================================================
# Main Sweep
# ============================================================


if __name__ == "__main__":
    n_mode = 3
    Ms = [1000, 2000, 4000, 8000, 16000]

    print("numba acceleration")
    print(" M      Iter    Setup(s)  Solve(s)  Eval(s)   Total(s)   RelL2       Linf")
    print("-"*80)

    for M in Ms:
        res = run_bem(M, n_mode)
        print(f"{res['M']:6d}  {res['iterations']:6d}  "
              f"{res['setup_time']:8.3f}  {res['solve_time']:8.3f}  "
              f"{res['eval_time']:8.3f}  {res['total_time']:9.3f}  "
              f"{res['relative_L2_error']:9.2e}  {res['Linf_error']:9.2e}")