import numpy as np
import scipy.linalg as la
import time
from numba import njit, prange

# ==============================================================================
# QUADRATURE DATA
# ==============================================================================

_gp1d = np.array([-0.94910791, -0.74153119, -0.40584515, 0.0, 0.40584515, 0.74153119, 0.94910791])
_gw1d = np.array([0.12948497, 0.27970539, 0.38183005, 0.41795918, 0.38183005, 0.27970539, 0.12948497])
GP1D = (_gp1d + 1.0) / 2.0
GW1D = _gw1d / 2.0

_alpha = 1.0 / 3.0
_beta1 = 0.470142064105115
_beta2 = 0.059715871789770
_gamma1 = 0.101286507323456
_gamma2 = 0.797426985353087

U_TRI = np.array([_alpha, _beta1, _beta2, _beta1, _gamma1, _gamma2, _gamma1])
V_TRI = np.array([_alpha, _beta1, _beta1, _beta2, _gamma1, _gamma1, _gamma2])
W_TRI = np.array([0.225, 0.132394152788506, 0.132394152788506, 0.132394152788506,
                  0.125939180544827, 0.125939180544827, 0.125939180544827])

# ==============================================================================
# MESH GENERATION (Unchanged)
# ==============================================================================

def generate_face_mesh(N, origin, u_vec, v_vec):
    u = np.linspace(0, 1, N+1)
    v = np.linspace(0, 1, N+1)
    uu, vv = np.meshgrid(u, v, indexing='ij')
    pts = origin + uu[..., None] * u_vec + vv[..., None] * v_vec
    pts = pts.reshape(-1, 3)

    idx = np.arange((N+1)**2).reshape(N+1, N+1)
    elems = []
    for i in range(N):
        for j in range(N):
            n1 = idx[i, j]
            n2 = idx[i+1, j]
            n3 = idx[i+1, j+1]
            n4 = idx[i, j+1]
            elems.append([n1, n2, n3])
            elems.append([n1, n3, n4])
    return pts, np.array(elems)

def generate_cube_mesh(N):
    faces = [
        (np.array([0, 1, 0]), np.array([1, 0, 0]), np.array([0, -1, 0])),
        (np.array([0, 0, 1]), np.array([1, 0, 0]), np.array([0, 1, 0])),
        (np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 0, 1])),
        (np.array([1, 1, 0]), np.array([-1, 0, 0]), np.array([0, 0, 1])),
        (np.array([0, 1, 0]), np.array([0, -1, 0]), np.array([0, 0, 1])),
        (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))
    ]

    all_pts, all_elems = [], []
    offset = 0
    for origin, u_vec, v_vec in faces:
        pts, elems = generate_face_mesh(N, origin, u_vec, v_vec)
        all_pts.append(pts)
        all_elems.append(elems + offset)
        offset += len(pts)

    all_pts = np.vstack(all_pts)
    all_elems = np.vstack(all_elems)

    unique_pts, inverse_indices = np.unique(all_pts, axis=0, return_inverse=True)
    unique_elems = inverse_indices[all_elems]

    Ne = len(unique_elems)
    centroids = np.zeros((Ne, 3))
    normals = np.zeros((Ne, 3))
    areas = np.zeros(Ne)

    for i in range(Ne):
        v0 = unique_pts[unique_elems[i, 0]]
        v1 = unique_pts[unique_elems[i, 1]]
        v2 = unique_pts[unique_elems[i, 2]]

        centroids[i] = (v0 + v1 + v2) / 3.0

        cross_prod = np.cross(v1 - v0, v2 - v0)
        area = 0.5 * np.linalg.norm(cross_prod)
        areas[i] = area
        normals[i] = cross_prod / (2.0 * area)

    return unique_pts, unique_elems, centroids, normals, areas

# ==============================================================================
# OPTIMIZED INTEGRATION AND ASSEMBLY
# ==============================================================================

@njit(fastmath=True, cache=True)
def duffy_sub(Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, gp, gw):
    # Scalarized cross product for area
    ABx, ABy, ABz = Bx - Ax, By - Ay, Bz - Az
    ACx, ACy, ACz = Cx - Ax, Cy - Ay, Cz - Az

    cpx = ABy * ACz - ABz * ACy
    cpy = ABz * ACx - ABx * ACz
    cpz = ABx * ACy - ABy * ACx

    area = 0.5 * np.sqrt(cpx*cpx + cpy*cpy + cpz*cpz)
    if area < 1e-14:
        return 0.0

    val = 0.0
    d1x, d1y, d1z = Bx - Ax, By - Ay, Bz - Az
    d2x, d2y, d2z = Cx - Bx, Cy - By, Cz - Bz

    for i in range(len(gp)):
        v = gp[i]
        w = gw[i]
        vx = d1x + v * d2x
        vy = d1y + v * d2y
        vz = d1z + v * d2z

        r = np.sqrt(vx*vx + vy*vy + vz*vz)
        val += w / r

    return val * (2.0 * area) / (4.0 * np.pi)

@njit(fastmath=True, cache=True)
def duffy_integrate(cx, cy, cz, v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, gp, gw):
    G_val = 0.0
    G_val += duffy_sub(cx, cy, cz, v0x, v0y, v0z, v1x, v1y, v1z, gp, gw)
    G_val += duffy_sub(cx, cy, cz, v1x, v1y, v1z, v2x, v2y, v2z, gp, gw)
    G_val += duffy_sub(cx, cy, cz, v2x, v2y, v2z, v0x, v0y, v0z, gp, gw)
    return G_val

@njit(fastmath=True, cache=True)
def standard_integrate(cx, cy, cz, v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, nx, ny, nz, area, u_tri, v_tri, w_tri):
    G_val = 0.0
    H_val = 0.0

    for i in range(len(u_tri)):
        u = u_tri[i]
        v = v_tri[i]
        w = w_tri[i]

        w_uv = 1.0 - u - v
        yx = v0x * w_uv + v1x * u + v2x * v
        yy = v0y * w_uv + v1y * u + v2y * v
        yz = v0z * w_uv + v1z * u + v2z * v

        rx = cx - yx
        ry = cy - yy
        rz = cz - yz

        r2 = rx*rx + ry*ry + rz*rz
        r = np.sqrt(r2)
        r3 = r * r2

        dot_prod = rx*nx + ry*ny + rz*nz

        G_val += w / r
        H_val += w * dot_prod / r3

    factor = area / (4.0 * np.pi)
    return G_val * factor, H_val * factor

@njit(parallel=True, fastmath=True, cache=True)
def assemble_system(centroids, vertices, elements, normals, areas, u_tri, v_tri, w_tri, gp1d, gw1d):
    Ne = len(centroids)
    H = np.zeros((Ne, Ne))
    G = np.zeros((Ne, Ne))

    for i in prange(Ne):
        cx, cy, cz = centroids[i, 0], centroids[i, 1], centroids[i, 2]

        for j in range(Ne):
            n0, n1, n2 = elements[j, 0], elements[j, 1], elements[j, 2]
            v0x, v0y, v0z = vertices[n0, 0], vertices[n0, 1], vertices[n0, 2]
            v1x, v1y, v1z = vertices[n1, 0], vertices[n1, 1], vertices[n1, 2]
            v2x, v2y, v2z = vertices[n2, 0], vertices[n2, 1], vertices[n2, 2]

            area = areas[j]
            nx, ny, nz = normals[j, 0], normals[j, 1], normals[j, 2]

            if i == j:
                H[i, j] = 0.0
                G[i, j] = duffy_integrate(cx, cy, cz, v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, gp1d, gw1d)
            else:
                g_val, h_val = standard_integrate(cx, cy, cz, v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, nx, ny, nz, area, u_tri, v_tri, w_tri)
                G[i, j] = g_val
                H[i, j] = h_val

    # Enforce rigid-body mode for H
    for i in prange(Ne):
        sum_H = 0.0
        for j in range(Ne):
            if i != j:
                sum_H += H[i, j]
        H[i, i] = -sum_H

    return H, G

@njit(parallel=True, fastmath=True, cache=True)
def evaluate_interior(pts, q_sol, u_sol, vertices, elements, normals, areas, u_tri, v_tri, w_tri):
    N_pts = len(pts)
    Ne = len(elements)
    u_int = np.zeros(N_pts)

    for i in prange(N_pts):
        cx, cy, cz = pts[i, 0], pts[i, 1], pts[i, 2]
        val = 0.0

        for j in range(Ne):
            n0, n1, n2 = elements[j, 0], elements[j, 1], elements[j, 2]
            v0x, v0y, v0z = vertices[n0, 0], vertices[n0, 1], vertices[n0, 2]
            v1x, v1y, v1z = vertices[n1, 0], vertices[n1, 1], vertices[n1, 2]
            v2x, v2y, v2z = vertices[n2, 0], vertices[n2, 1], vertices[n2, 2]

            area = areas[j]
            nx, ny, nz = normals[j, 0], normals[j, 1], normals[j, 2]

            g_val, h_val = standard_integrate(cx, cy, cz, v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, nx, ny, nz, area, u_tri, v_tri, w_tri)
            val += g_val * q_sol[j] - h_val * u_sol[j]

        u_int[i] = val
    return u_int

# ==============================================================================
# MAIN SOLVER
# ==============================================================================

def run_solver(N):
    t0 = time.time()
    vertices, elements, centroids, normals, areas = generate_cube_mesh(N)
    Ne = len(elements)

    # Boundary Conditions
    bc_type = np.zeros(Ne, dtype=np.int32)
    bc_val = np.zeros(Ne)

    for i in range(Ne):
        cx, cy, cz = centroids[i]
        if abs(cx) < 1e-7:
            bc_type[i] = 0; bc_val[i] = cy * cz
        elif abs(cy) < 1e-7:
            bc_type[i] = 0; bc_val[i] = cz * cx
        elif abs(cz) < 1e-7:
            bc_type[i] = 0; bc_val[i] = cx * cy
        elif abs(cx - 1.0) < 1e-7:
            bc_type[i] = 1; bc_val[i] = cy + cz
        elif abs(cy - 1.0) < 1e-7:
            bc_type[i] = 1; bc_val[i] = cx + cz
        elif abs(cz - 1.0) < 1e-7:
            bc_type[i] = 1; bc_val[i] = cx + cy

    H, G = assemble_system(centroids, vertices, elements, normals, areas, U_TRI, V_TRI, W_TRI, GP1D, GW1D)

    A = np.zeros((Ne, Ne))
    b = np.zeros(Ne)

    for j in range(Ne):
        if bc_type[j] == 0:
            A[:, j] = -G[:, j]
            b -= H[:, j] * bc_val[j]
        else:
            A[:, j] = H[:, j]
            b += G[:, j] * bc_val[j]

    t1 = time.time()

    # Use scipy solver with overwrite to save memory and speed up
    x_vec = la.solve(A, b, overwrite_a=True, overwrite_b=True)
    t2 = time.time()

    u_sol = np.zeros(Ne)
    q_sol = np.zeros(Ne)
    for j in range(Ne):
        if bc_type[j] == 0:
            u_sol[j] = bc_val[j]
            q_sol[j] = x_vec[j]
        else:
            q_sol[j] = bc_val[j]
            u_sol[j] = x_vec[j]

    # Interior Evaluation
    gx = np.linspace(0.2, 0.8, 10)
    gy = np.linspace(0.2, 0.8, 10)
    gz = np.linspace(0.2, 0.8, 10)
    XX, YY, ZZ = np.meshgrid(gx, gy, gz, indexing='ij')
    grid_pts = np.vstack([XX.ravel(), YY.ravel(), ZZ.ravel()]).T

    u_int = evaluate_interior(grid_pts, q_sol, u_sol, vertices, elements, normals, areas, U_TRI, V_TRI, W_TRI)
    t3 = time.time()

    u_exact = grid_pts[:, 0] * grid_pts[:, 1] + grid_pts[:, 1] * grid_pts[:, 2] + grid_pts[:, 2] * grid_pts[:, 0]
    err_l2 = np.linalg.norm(u_int - u_exact) / np.linalg.norm(u_exact)

    times = {
        'setup': t1 - t0,
        'solve': t2 - t1,
        'eval': t3 - t2,
        'total': t3 - t0
    }

    return Ne, err_l2, times

if __name__ == "__main__":
    Ns = [8, 16, 32]
    results = []

    # Warmup Numba (JIT compilation)
    _ = run_solver(2)

    print("-" * 80)
    print(f"{'N':<5} | {'Ne':<6} | {'Rel L2 Error':<15} | {'Setup (s)':<10} | {'Solve (s)':<10} | {'Eval (s)':<10} | {'Total (s)':<10}")
    print("-" * 80)

    for N in Ns:
        Ne, err_l2, times = run_solver(N)
        results.append((Ne, err_l2))
        print(f"{N:<5} | {Ne:<6} | {err_l2:<15.6e} | {times['setup']:<10.4f} | {times['solve']:<10.4f} | {times['eval']:<10.4f} | {times['total']:<10.4f}")

    print("-" * 80)

    Nes = np.array([r[0] for r in results])
    errs = np.array([r[1] for r in results])
    h = np.sqrt(1.0 / Nes)

    slope, intercept = np.polyfit(np.log(h), np.log(errs), 1)

    print(f"\nConvergence Analysis:")
    print(f"Computed Slope: {slope:.4f}")
    print(f"Expected Slope: ~1.0000 (O(h))")