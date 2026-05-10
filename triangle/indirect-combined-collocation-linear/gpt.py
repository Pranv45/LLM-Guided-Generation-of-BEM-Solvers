import numpy as np
import time
from numpy.polynomial.legendre import leggauss
from scipy.sparse.linalg import gmres, LinearOperator
from numba import njit

# ============================================================
# Exact solution
# ============================================================

def u_exact(x, y):
    return np.sin(np.pi * x) * np.cosh(np.pi * y)

# ============================================================
# Cosine spacing
# ============================================================

def cosine_nodes(a, b, N):
    theta = np.linspace(0, np.pi, N+1)
    s = 0.5 * (1 - np.cos(theta))
    return a + (b - a) * s

# ============================================================
# Build boundary mesh (linear elements, node-based DOFs)
# ============================================================

def build_boundary(N):

    # Edge 1: (0,0) -> (1,0)
    x1 = cosine_nodes(0.0, 1.0, N)
    y1 = np.zeros_like(x1)

    # Edge 2: (1,0) -> (0,1)
    s2 = cosine_nodes(0.0, 1.0, N)
    x2 = 1.0 - s2
    y2 = s2

    # Edge 3: (0,1) -> (0,0)
    y3 = cosine_nodes(1.0, 0.0, N)
    x3 = np.zeros_like(y3)

    # Concatenate nodes without duplicating corners
    xs = np.concatenate([x1[:-1], x2[:-1], x3[:-1]])
    ys = np.concatenate([y1[:-1], y2[:-1], y3[:-1]])
    nodes = np.column_stack((xs, ys))

    M = nodes.shape[0]
    elems = np.array([(i, (i+1) % M) for i in range(M)], dtype=np.int64)

    # Element geometry
    Apts = nodes[elems[:,0]]
    Bpts = nodes[elems[:,1]]
    tx = Bpts[:,0] - Apts[:,0]
    ty = Bpts[:,1] - Apts[:,1]
    lengths = np.sqrt(tx*tx + ty*ty)
    nx = ty / lengths
    ny = -tx / lengths

    return nodes, elems, Apts, Bpts, nx, ny, lengths

# ============================================================
# Kernel
# ============================================================

@njit
def dGdn(dx, dy, r, nx, ny):
    return -(1.0/(2.0*np.pi)) * (dx*nx + dy*ny) / (r*r)

# ============================================================
# Matrix-free operator
# ============================================================

@njit
def matvec_numba(mu, nodes, elems, Apts, Bpts, nx, ny, lengths,
                 gp, gw):

    M = nodes.shape[0]
    nq = gp.shape[0]
    result = np.zeros(M)

    for i in range(M):

        xi = nodes[i,0]
        yi = nodes[i,1]

        val = 0.5 * mu[i]

        for e in range(M):

            n1 = elems[e,0]
            n2 = elems[e,1]

            x1 = Apts[e,0]
            y1 = Apts[e,1]
            x2 = Bpts[e,0]
            y2 = Bpts[e,1]

            L = lengths[e]
            nxj = nx[e]
            nyj = ny[e]

            acc1 = 0.0
            acc2 = 0.0

            for q in range(nq):

                s = gp[q]
                w = gw[q]

                phi1 = 0.5*(1.0 - s)
                phi2 = 0.5*(1.0 + s)

                xq = phi1*x1 + phi2*x2
                yq = phi1*y1 + phi2*y2

                dx = xi - xq
                dy = yi - yq
                r = np.sqrt(dx*dx + dy*dy)

                if r > 1e-14:
                    kern = dGdn(dx, dy, r, nxj, nyj)
                    acc1 += kern * phi1 * w
                    acc2 += kern * phi2 * w

            val += (acc1 * mu[n1] + acc2 * mu[n2]) * (L/2.0)

        result[i] = val

    return result

# ============================================================
# Interior evaluation
# ============================================================

@njit
def evaluate_interior(mu, X, Y, elems, Apts, Bpts,
                      nx, ny, lengths, gp, gw):

    npts = X.shape[0]
    M = lengths.shape[0]
    nq = gp.shape[0]
    u = np.zeros(npts)

    for p in range(npts):

        xp = X[p]
        yp = Y[p]
        val = 0.0

        for e in range(M):

            n1 = elems[e,0]
            n2 = elems[e,1]

            x1 = Apts[e,0]
            y1 = Apts[e,1]
            x2 = Bpts[e,0]
            y2 = Bpts[e,1]

            L = lengths[e]
            nxj = nx[e]
            nyj = ny[e]

            acc1 = 0.0
            acc2 = 0.0

            for q in range(nq):

                s = gp[q]
                w = gw[q]

                phi1 = 0.5*(1.0 - s)
                phi2 = 0.5*(1.0 + s)

                xq = phi1*x1 + phi2*x2
                yq = phi1*y1 + phi2*y2

                dx = xp - xq
                dy = yp - yq
                r = np.sqrt(dx*dx + dy*dy)

                if r > 1e-14:
                    kern = dGdn(dx, dy, r, nxj, nyj)
                    acc1 += kern * phi1 * w
                    acc2 += kern * phi2 * w

            val += (acc1 * mu[n1] + acc2 * mu[n2]) * (L/2.0)

        u[p] = val

    return u

# ============================================================
# Main
# ============================================================

gp, gw = leggauss(8)
N_values = [25, 50, 100, 200, 400]

print("N   DOFs   GMRES_it   RelL2     Setup(s) Solve(s) Eval(s) Total(s)")

for N in N_values:

    t0 = time.time()

    nodes, elems, Apts, Bpts, nx, ny, lengths = build_boundary(N)
    M = nodes.shape[0]

    f = u_exact(nodes[:,0], nodes[:,1])

    t_setup = time.time()

    def mv(v):
        return matvec_numba(v, nodes, elems, Apts, Bpts,
                            nx, ny, lengths, gp, gw)

    Aop = LinearOperator((M, M), matvec=mv)

    it_count = [0]
    def cb(rk):
        it_count[0] += 1

    mu, info = gmres(Aop, f, atol=1e-8, callback=cb)

    t_solve = time.time()

    # Interior grid
    ngrid = 60
    x = np.linspace(0.02, 0.98, ngrid)
    y = np.linspace(0.02, 0.98, ngrid)
    XX, YY = np.meshgrid(x, y)
    mask = XX + YY < 1.0

    Xf = XX[mask]
    Yf = YY[mask]

    u_num = evaluate_interior(mu, Xf, Yf, elems,
                              Apts, Bpts, nx, ny,
                              lengths, gp, gw)

    u_ex = u_exact(Xf, Yf)
    relL2 = np.linalg.norm(u_num - u_ex) / np.linalg.norm(u_ex)

    t_eval = time.time()

    print(f"{N:<4}{M:<7}{it_count[0]:<11}{relL2:8.3e}"
          f"{(t_setup-t0):9.3f}{(t_solve-t_setup):9.3f}"
          f"{(t_eval-t_solve):9.3f}{(t_eval-t0):9.3f}")