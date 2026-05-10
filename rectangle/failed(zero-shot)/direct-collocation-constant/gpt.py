import numpy as np
import time
from scipy.linalg import solve

# ============================================================
# Problem setup
# ============================================================

def u_exact(x, y):
    return (np.sinh(np.pi * x) / np.sinh(2 * np.pi)) * np.cos(np.pi * y)

def q_exact(x, y, nx, ny):
    ux = (np.pi * np.cosh(np.pi * x) / np.sinh(2 * np.pi)) * np.cos(np.pi * y)
    uy = -(np.sinh(np.pi * x) / np.sinh(2 * np.pi)) * np.pi * np.sin(np.pi * y)
    return ux * nx + uy * ny

# ============================================================
# Boundary discretization
# ============================================================

def discretize(N):
    panels = []
    # bottom (Neumann)
    xs = np.linspace(0, 2, N+1)
    for i in range(N):
        x1, x2 = xs[i], xs[i+1]
        mid = np.array([(x1+x2)/2, 0.0])
        length = x2 - x1
        panels.append((mid, length, 0.0, -1.0, 'N'))
    # right (Dirichlet)
    ys = np.linspace(0, 1, N+1)
    for i in range(N):
        y1, y2 = ys[i], ys[i+1]
        mid = np.array([2.0, (y1+y2)/2])
        length = y2 - y1
        panels.append((mid, length, 1.0, 0.0, 'D'))
    # top (Neumann)
    xs = np.linspace(2, 0, N+1)
    for i in range(N):
        x1, x2 = xs[i], xs[i+1]
        mid = np.array([(x1+x2)/2, 1.0])
        length = abs(x2 - x1)
        panels.append((mid, length, 0.0, 1.0, 'N'))
    # left (Dirichlet)
    ys = np.linspace(1, 0, N+1)
    for i in range(N):
        y1, y2 = ys[i], ys[i+1]
        mid = np.array([0.0, (y1+y2)/2])
        length = abs(y2 - y1)
        panels.append((mid, length, -1.0, 0.0, 'D'))
    return panels

# ============================================================
# Kernel functions
# ============================================================

def G_kernel(r):
    return -(1.0/(2*np.pi)) * np.log(r)

def H_kernel(dx, dy, r, nx, ny):
    return -(1.0/(2*np.pi)) * (dx*nx + dy*ny) / (r**2)

# ============================================================
# Main BEM solve
# ============================================================

def solve_bem(N):
    t0 = time.time()
    panels = discretize(N)
    M = len(panels)

    mids = np.array([p[0] for p in panels])
    lengths = np.array([p[1] for p in panels])
    normals = np.array([[p[2], p[3]] for p in panels])
    types = [p[4] for p in panels]

    u_known = np.zeros(M)
    q_known = np.zeros(M)
    unknown_idx = []
    is_q_unknown = []

    for i, (mid, length, nx, ny, bc) in enumerate(panels):
        x, y = mid
        if bc == 'D':
            u_known[i] = u_exact(x, y)
            unknown_idx.append(i)
            is_q_unknown.append(True)
        else:
            q_known[i] = 0.0
            unknown_idx.append(i)
            is_q_unknown.append(False)

    A = np.zeros((M, M))
    rhs = np.zeros(M)

    for i in range(M):
        xi = mids[i]
        for j in range(M):
            xj = mids[j]
            dx = xi[0] - xj[0]
            dy = xi[1] - xj[1]
            r = np.hypot(dx, dy)
            if i == j:
                G = -(lengths[j]/(2*np.pi)) * (np.log(lengths[j]/2) - 1)
                H = 0.0
            else:
                G = G_kernel(r) * lengths[j]
                H = H_kernel(dx, dy, r, normals[j,0], normals[j,1]) * lengths[j]

            if is_q_unknown[j]:
                A[i,j] = -G
                rhs[i] += H * u_known[j]
            else:
                A[i,j] = H
                rhs[i] += -G * q_known[j]

        if types[i] == 'D':
            rhs[i] += 0.5 * u_known[i]
        else:
            A[i,i] -= 0.5

    t_assembly = time.time()

    sol = solve(A, rhs)

    t_solve = time.time()

    u_vals = np.zeros(M)
    q_vals = np.zeros(M)
    for i in range(M):
        if is_q_unknown[i]:
            u_vals[i] = u_known[i]
            q_vals[i] = sol[i]
        else:
            u_vals[i] = sol[i]
            q_vals[i] = q_known[i]

    # Interior evaluation
    nxg, nyg = 40, 20
    xs = np.linspace(0.05, 1.95, nxg)
    ys = np.linspace(0.05, 0.95, nyg)
    XX, YY = np.meshgrid(xs, ys)
    u_num = np.zeros_like(XX)

    for i in range(nxg):
        for j in range(nyg):
            x = np.array([XX[j,i], YY[j,i]])
            val = 0.0
            for k in range(M):
                dx = x[0] - mids[k,0]
                dy = x[1] - mids[k,1]
                r = np.hypot(dx, dy)
                G = G_kernel(r) * lengths[k]
                H = H_kernel(dx, dy, r, normals[k,0], normals[k,1]) * lengths[k]
                val += H * u_vals[k] - G * q_vals[k]
            u_num[j,i] = val

    u_ex = u_exact(XX, YY)
    rel_L2 = np.linalg.norm(u_num - u_ex) / np.linalg.norm(u_ex)

    t_eval = time.time()
    return M, t_assembly - t0, t_solve - t_assembly, t_eval - t_solve, t_eval - t0, rel_L2

# ============================================================
# Refinement study
# ============================================================

Ns = [20, 40, 80, 160, 320, 640]
prev_err = None

print("N  DOFs  Assem(s)  Solve(s)  Eval(s)  Total(s)  RelL2     Rate")
for N in Ns:
    dofs, ta, ts, te, tt, err = solve_bem(N)
    rate = np.log(prev_err/err)/np.log(2) if prev_err is not None else np.nan
    print(f"{N:<3} {dofs:<5} {ta:8.3f} {ts:8.3f} {te:8.3f} {tt:8.3f} {err:8.2e} {rate:6.2f}")
    prev_err = err