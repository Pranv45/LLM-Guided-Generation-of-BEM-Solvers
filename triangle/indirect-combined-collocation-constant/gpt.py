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
    s = 0.5*(1 - np.cos(theta))
    return a + (b-a)*s

# ============================================================
# Boundary discretization of triangle
# ============================================================

def build_boundary(N):

    # Edge 1: (0,0) -> (1,0)
    x1 = cosine_nodes(0, 1, N)
    y1 = np.zeros_like(x1)

    # Edge 2: (1,0) -> (0,1)
    s2 = cosine_nodes(0, 1, N)
    x2 = 1 - s2
    y2 = s2

    # Edge 3: (0,1) -> (0,0)
    y3 = cosine_nodes(1, 0, N)
    x3 = np.zeros_like(y3)

    xs = np.concatenate([x1[:-1], x2[:-1], x3[:-1]])
    ys = np.concatenate([y1[:-1], y2[:-1], y3[:-1]])

    xe = np.concatenate([x1[1:], x2[1:], y3[1:]*0])
    ye = np.concatenate([y1[1:], y2[1:], y3[1:]])

    Apts = np.column_stack((xs, ys))
    Bpts = np.column_stack((xe, ye))

    centers = 0.5*(Apts + Bpts)
    tx = Bpts[:,0] - Apts[:,0]
    ty = Bpts[:,1] - Apts[:,1]
    lengths = np.sqrt(tx**2 + ty**2)

    nx = ty / lengths
    ny = -tx / lengths

    return Apts, Bpts, centers, nx, ny, lengths

# ============================================================
# Kernels
# ============================================================

@njit
def dGdn(dx, dy, r, nx, ny):
    return -(1.0/(2*np.pi))*(dx*nx + dy*ny)/(r*r)

@njit
def G(r):
    return -(1.0/(2*np.pi))*np.log(r)

# ============================================================
# Matrix-free operator
# ============================================================

@njit
def matvec_numba(mu, centers, Apts, Bpts, nx, ny, lengths,
                 gp, gw, alpha):

    M = centers.shape[0]
    nq = gp.shape[0]
    result = np.zeros(M)

    for i in range(M):

        xi = centers[i,0]
        yi = centers[i,1]

        val = 0.5 * mu[i]

        for j in range(M):

            x1 = Apts[j,0]
            y1 = Apts[j,1]
            x2 = Bpts[j,0]
            y2 = Bpts[j,1]

            L = lengths[j]
            nxj = nx[j]
            nyj = ny[j]

            acc = 0.0

            for q in range(nq):

                s = gp[q]
                w = gw[q]

                xq = 0.5*((1-s)*x1 + (1+s)*x2)
                yq = 0.5*((1-s)*y1 + (1+s)*y2)

                dx = xi - xq
                dy = yi - yq
                r = np.sqrt(dx*dx + dy*dy)

                if r > 1e-14:
                    acc += (dGdn(dx,dy,r,nxj,nyj) + alpha*G(r)) * w

            val += acc * (L/2) * mu[j]

        result[i] = val

    return result

# ============================================================
# Interior evaluation
# ============================================================

@njit
def evaluate_interior(mu, X, Y, Apts, Bpts, nx, ny, lengths,
                      gp, gw, alpha):

    npts = X.shape[0]
    M = lengths.shape[0]
    nq = gp.shape[0]
    u = np.zeros(npts)

    for p in range(npts):

        xp = X[p]
        yp = Y[p]

        val = 0.0

        for j in range(M):

            x1 = Apts[j,0]
            y1 = Apts[j,1]
            x2 = Bpts[j,0]
            y2 = Bpts[j,1]

            L = lengths[j]
            nxj = nx[j]
            nyj = ny[j]

            acc = 0.0

            for q in range(nq):

                s = gp[q]
                w = gw[q]

                xq = 0.5*((1-s)*x1 + (1+s)*x2)
                yq = 0.5*((1-s)*y1 + (1+s)*y2)

                dx = xp - xq
                dy = yp - yq
                r = np.sqrt(dx*dx + dy*dy)

                if r > 1e-14:
                    acc += (dGdn(dx,dy,r,nxj,nyj) + alpha*G(r)) * w

            val += acc * (L/2) * mu[j]

        u[p] = val

    return u

# ============================================================
# Main solver
# ============================================================

alpha = 0.1
gp, gw = leggauss(8)

N_values = [25, 50, 100, 200, 400]

print("N   DOFs   GMRES_it   RelL2     Setup(s) Solve(s) Eval(s) Total(s)")

for N in N_values:

    t0 = time.time()

    Apts, Bpts, centers, nx, ny, lengths = build_boundary(N)
    M = centers.shape[0]

    f = u_exact(centers[:,0], centers[:,1])

    t_setup = time.time()

    def matvec(v):
        return matvec_numba(v, centers, Apts, Bpts,
                            nx, ny, lengths,
                            gp, gw, alpha)

    Aop = LinearOperator((M,M), matvec=matvec)

    iter_count = [0]
    def callback(rk):
        iter_count[0] += 1

    mu, info = gmres(Aop, f, atol=1e-10, callback=callback, callback_type='legacy')

    t_solve = time.time()

    # Interior grid
    ngrid = 60
    x = np.linspace(0.02, 0.98, ngrid)
    y = np.linspace(0.02, 0.98, ngrid)
    XX, YY = np.meshgrid(x,y)

    mask = XX + YY < 1.0
    Xf = XX[mask]
    Yf = YY[mask]

    u_num = evaluate_interior(mu, Xf, Yf,
                              Apts, Bpts, nx, ny, lengths,
                              gp, gw, alpha)

    u_ex = u_exact(Xf, Yf)
    relL2 = np.linalg.norm(u_num - u_ex) / np.linalg.norm(u_ex)

    t_eval = time.time()

    print(f"{N:<4}{M:<7}{iter_count[0]:<11}{relL2:8.3e}"
          f"{(t_setup-t0):9.3f}{(t_solve-t_setup):9.3f}"
          f"{(t_eval-t_solve):9.3f}{(t_eval-t0):9.3f}")