import numpy as np
import time
import gc
import os
import tempfile
import scipy.sparse.linalg as spla
from numba import njit, prange

def generate_bumpy_mesh(N):
    u = np.linspace(-1, 1, N+1)
    v = np.linspace(-1, 1, N+1)
    uu, vv = np.meshgrid(u, v)
    u_flat = uu.flatten()
    v_flat = vv.flatten()

    nodes_list = []
    elems_list = []
    node_offset = 0

    for face_id in range(6):
        if face_id == 0:   X, Y, Z = np.ones_like(u_flat), u_flat, v_flat
        elif face_id == 1: X, Y, Z = -np.ones_like(u_flat), v_flat, u_flat
        elif face_id == 2: X, Y, Z = u_flat, np.ones_like(u_flat), v_flat
        elif face_id == 3: X, Y, Z = v_flat, -np.ones_like(u_flat), u_flat
        elif face_id == 4: X, Y, Z = u_flat, v_flat, np.ones_like(u_flat)
        elif face_id == 5: X, Y, Z = v_flat, u_flat, -np.ones_like(u_flat)

        R = np.sqrt(X**2 + Y**2 + Z**2)
        X_sph, Y_sph, Z_sph = X/R, Y/R, Z/R
        Z_sph = np.clip(Z_sph, -1.0, 1.0)

        theta = np.arccos(Z_sph)
        phi = np.arctan2(Y_sph, X_sph)
        r = 1.5 + 0.3 * np.sin(4*theta) * np.cos(5*phi)

        X_f = r * np.sin(theta) * np.cos(phi)
        Y_f = r * np.sin(theta) * np.sin(phi)
        Z_f = r * np.cos(theta)

        face_nodes = np.vstack((X_f, Y_f, Z_f)).T
        nodes_list.append(face_nodes)

        for i in range(N):
            for j in range(N):
                n0 = i*(N+1) + j
                n1 = i*(N+1) + j + 1
                n2 = (i+1)*(N+1) + j
                n3 = (i+1)*(N+1) + j + 1

                elems_list.append([node_offset + n0, node_offset + n1, node_offset + n2])
                elems_list.append([node_offset + n1, node_offset + n3, node_offset + n2])

        node_offset += len(face_nodes)

    nodes = np.vstack(nodes_list)
    elems = np.array(elems_list, dtype=np.int32)
    return nodes, elems

def compute_geometry(nodes, elems):
    v0 = nodes[elems[:, 0]]
    v1 = nodes[elems[:, 1]]
    v2 = nodes[elems[:, 2]]

    centroids = (v0 + v1 + v2) / 3.0
    vec1 = v1 - v0
    vec2 = v2 - v0
    cross = np.cross(vec1, vec2)

    areas = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / (2.0 * areas[:, None])

    dots = np.sum(normals * centroids, axis=1)
    flips = dots < 0
    normals[flips] *= -1.0

    temp = elems[flips, 1].copy()
    elems[flips, 1] = elems[flips, 2]
    elems[flips, 2] = temp

    return centroids, areas, normals, elems

def assign_mixed_bcs(centroids, normals):
    Ne = centroids.shape[0]
    bc_type = np.zeros(Ne, dtype=np.int32)
    bc_val = np.zeros(Ne)

    X, Y, Z = centroids[:, 0], centroids[:, 1], centroids[:, 2]

    u_exact = np.sinh(X)*np.sin(Y) + np.cosh(Y)*np.cos(Z)
    grad_x = np.cosh(X)*np.sin(Y)
    grad_y = np.sinh(X)*np.cos(Y) + np.sinh(Y)*np.cos(Z)
    grad_z = -np.cosh(Y)*np.sin(Z)

    q_exact = grad_x*normals[:, 0] + grad_y*normals[:, 1] + grad_z*normals[:, 2]

    dirichlet_mask = X > 0
    bc_type[dirichlet_mask] = 0
    bc_val[dirichlet_mask] = u_exact[dirichlet_mask]

    neumann_mask = X <= 0
    bc_type[neumann_mask] = 1
    bc_val[neumann_mask] = q_exact[neumann_mask]

    return bc_type, bc_val

def precompute_integrations(nodes, elems, areas, gp_near, gw_near):
    Ne = elems.shape[0]
    dun_bary = np.array([
        [1/3, 1/3, 1/3],
        [0.470142064105115, 0.470142064105115, 0.059715871789770],
        [0.470142064105115, 0.059715871789770, 0.470142064105115],
        [0.059715871789770, 0.470142064105115, 0.470142064105115],
        [0.101286507323456, 0.101286507323456, 0.797426985353087],
        [0.101286507323456, 0.797426985353087, 0.101286507323456],
        [0.797426985353087, 0.101286507323456, 0.101286507323456]
    ])
    dun_weight = np.array([
        0.225000000000000, 0.132394152788506, 0.132394152788506, 0.132394152788506,
        0.125939180544827, 0.125939180544827, 0.125939180544827
    ])

    dun_pts = np.zeros((Ne, 7, 3))
    dun_wts = np.zeros((Ne, 7))

    N_near = len(gp_near)
    near_pts = np.zeros((Ne, N_near * N_near, 3))
    near_wts = np.zeros((Ne, N_near * N_near))

    gu = (gp_near + 1.0) / 2.0
    gwu = gw_near / 2.0

    v0 = nodes[elems[:, 0]]
    v1 = nodes[elems[:, 1]]
    v2 = nodes[elems[:, 2]]

    for i in range(Ne):
        for k in range(7):
            dun_pts[i, k, :] = dun_bary[k, 0] * v0[i] + dun_bary[k, 1] * v1[i] + dun_bary[k, 2] * v2[i]
            dun_wts[i, k] = dun_weight[k] * areas[i]

        idx = 0
        for u_idx in range(N_near):
            for v_idx in range(N_near):
                u = gu[u_idx]
                v = gu[v_idx]
                xi1 = u
                xi2 = v * (1.0 - u)
                xi0 = 1.0 - xi1 - xi2

                near_pts[i, idx, 0] = xi0 * v0[i, 0] + xi1 * v1[i, 0] + xi2 * v2[i, 0]
                near_pts[i, idx, 1] = xi0 * v0[i, 1] + xi1 * v1[i, 1] + xi2 * v2[i, 1]
                near_pts[i, idx, 2] = xi0 * v0[i, 2] + xi1 * v1[i, 2] + xi2 * v2[i, 2]
                near_wts[i, idx] = gwu[u_idx] * gwu[v_idx] * (1.0 - u) * 2.0 * areas[i]
                idx += 1

    return dun_pts, dun_wts, near_pts, near_wts

@njit(parallel=True, fastmath=True)
def assemble_system(centroids, areas, nodes, elems, dun_pts, dun_wts, near_pts, near_wts, gauss_pts_1d, gauss_wts_1d, normals, bc_type, bc_val, A, b):
    Ne = centroids.shape[0]

    gp = (gauss_pts_1d + 1.0) / 2.0
    gw = gauss_wts_1d / 2.0
    N_g = len(gauss_pts_1d)
    N_near_total = near_pts.shape[1]

    for i in prange(Ne):
        cx = centroids[i, 0]
        cy = centroids[i, 1]
        cz = centroids[i, 2]

        b_i = 0.0
        H_ii = 0.0
        G_ii = 0.0

        for j in range(Ne):
            h_val = 0.0
            g_val = 0.0

            if i == j:
                v0x, v0y, v0z = nodes[elems[j, 0], 0], nodes[elems[j, 0], 1], nodes[elems[j, 0], 2]
                v1x, v1y, v1z = nodes[elems[j, 1], 0], nodes[elems[j, 1], 1], nodes[elems[j, 1], 2]
                v2x, v2y, v2z = nodes[elems[j, 2], 0], nodes[elems[j, 2], 1], nodes[elems[j, 2], 2]

                for sub in range(3):
                    if sub == 0:
                        p1x, p1y, p1z = v0x, v0y, v0z
                        p2x, p2y, p2z = v1x, v1y, v1z
                    elif sub == 1:
                        p1x, p1y, p1z = v1x, v1y, v1z
                        p2x, p2y, p2z = v2x, v2y, v2z
                    else:
                        p1x, p1y, p1z = v2x, v2y, v2z
                        p2x, p2y, p2z = v0x, v0y, v0z

                    cpx = (p1y - cy)*(p2z - cz) - (p1z - cz)*(p2y - cy)
                    cpy = (p1z - cz)*(p2x - cx) - (p1x - cx)*(p2z - cz)
                    cpz = (p1x - cx)*(p2y - cy) - (p1y - cy)*(p2x - cx)
                    two_asub = np.sqrt(cpx*cpx + cpy*cpy + cpz*cpz)

                    for u_idx in range(N_g):
                        for v_idx in range(N_g):
                            eta = gp[v_idx]
                            weight = gw[u_idx] * gw[v_idx]

                            dx = p1x - cx + eta * (p2x - p1x)
                            dy = p1y - cy + eta * (p2y - p1y)
                            dz = p1z - cz + eta * (p2z - p1z)

                            r = np.sqrt(dx*dx + dy*dy + dz*dz)
                            if r > 1e-14:
                                g_val += weight * two_asub / (4.0 * np.pi * r)

                G_ii = g_val

            else:
                nx, ny, nz = normals[j, 0], normals[j, 1], normals[j, 2]

                sx, sy, sz = centroids[j, 0], centroids[j, 1], centroids[j, 2]
                dist_sq = (cx - sx)**2 + (cy - sy)**2 + (cz - sz)**2

                if dist_sq < 20.0 * areas[j]:
                    for k in range(N_near_total):
                        qx = near_pts[j, k, 0]
                        qy = near_pts[j, k, 1]
                        qz = near_pts[j, k, 2]
                        w  = near_wts[j, k]

                        rx = cx - qx
                        ry = cy - qy
                        rz = cz - qz
                        r2 = rx*rx + ry*ry + rz*rz
                        r = np.sqrt(r2)

                        rdotn = rx*nx + ry*ny + rz*nz

                        h_val += w * (rdotn / (4.0 * np.pi * r2 * r))
                        g_val += w * (1.0 / (4.0 * np.pi * r))
                else:
                    for k in range(7):
                        qx = dun_pts[j, k, 0]
                        qy = dun_pts[j, k, 1]
                        qz = dun_pts[j, k, 2]
                        w  = dun_wts[j, k]

                        rx = cx - qx
                        ry = cy - qy
                        rz = cz - qz
                        r2 = rx*rx + ry*ry + rz*rz
                        r = np.sqrt(r2)

                        rdotn = rx*nx + ry*ny + rz*nz

                        h_val += w * (rdotn / (4.0 * np.pi * r2 * r))
                        g_val += w * (1.0 / (4.0 * np.pi * r))

                H_ii -= h_val

                if bc_type[j] == 0:
                    A[i, j] = -g_val
                    b_i -= h_val * bc_val[j]
                else:
                    A[i, j] = h_val
                    b_i += g_val * bc_val[j]

        if bc_type[i] == 0:
            A[i, i] = -G_ii
            b_i -= H_ii * bc_val[i]
        else:
            A[i, i] = H_ii
            b_i += G_ii * bc_val[i]

        b[i] = b_i

@njit(parallel=True, fastmath=True)
def evaluate_interior(eval_pts, u_full, q_full, dun_pts, dun_wts, normals):
    N_eval = eval_pts.shape[0]
    Ne = u_full.shape[0]
    u_int = np.zeros(N_eval, dtype=np.float64)

    for i in prange(N_eval):
        cx = eval_pts[i, 0]
        cy = eval_pts[i, 1]
        cz = eval_pts[i, 2]
        val = 0.0

        for j in range(Ne):
            h_val = 0.0
            g_val = 0.0
            nx, ny, nz = normals[j, 0], normals[j, 1], normals[j, 2]

            for k in range(7):
                qx = dun_pts[j, k, 0]
                qy = dun_pts[j, k, 1]
                qz = dun_pts[j, k, 2]
                w  = dun_wts[j, k]

                rx = cx - qx
                ry = cy - qy
                rz = cz - qz
                r2 = rx*rx + ry*ry + rz*rz
                r = np.sqrt(r2)

                rdotn = rx*nx + ry*ny + rz*nz

                h_val += w * (rdotn / (4.0 * np.pi * r2 * r))
                g_val += w * (1.0 / (4.0 * np.pi * r))

            val += g_val * q_full[j] - h_val * u_full[j]

        u_int[i] = val

    return u_int

class OutOfCoreMatrix(spla.LinearOperator):
    """
    Wraps the memory-mapped matrix into a SciPy Linear Operator to stream
    matrix-vector multiplications dynamically from disk during GMRES.
    """
    def __init__(self, memmap_matrix, shape):
        self.A = memmap_matrix
        self.shape = shape
        # FIX: SciPy requires an instantiated dtype object, not the raw class
        self.dtype = np.dtype('float64')

    def _matvec(self, v):
        res = np.zeros(self.shape[0], dtype=self.dtype)
        chunk_size = 2048 # Adjust this based on available RAM
        for i in range(0, self.shape[0], chunk_size):
            end = min(i + chunk_size, self.shape[0])
            # Pulls a small chunk of rows from disk and computes the dot product
            res[i:end] = self.A[i:end, :].dot(v)
        return res

def main():
    print("N    | Ne      | Rel L2 Error   | Setup (s) | Solve (s) | Eval (s) | Total (s)")
    results = []

    gp_self, gw_self = np.polynomial.legendre.leggauss(7)
    gp_near, gw_near = np.polynomial.legendre.leggauss(12)

    # Pre-compile JIT routines
    _n, _e = generate_bumpy_mesh(1)
    _c, _a, _no, _e = compute_geometry(_n, _e)
    _dp, _dw, _np_pts, _nw = precompute_integrations(_n, _e, _a, gp_near, gw_near)
    _bt, _bv = assign_mixed_bcs(_c, _no)

    _A = np.zeros((len(_c), len(_c)), dtype=np.float64)
    _b = np.zeros(len(_c), dtype=np.float64)
    assemble_system(_c, _a, _n, _e, _dp, _dw, _np_pts, _nw, gp_self, gw_self, _no, _bt, _bv, _A, _b)

    _eval_pts = np.ascontiguousarray(np.array([[0.0, 0.0, 0.0]]))
    _uf, _qf = np.zeros(len(_c)), np.zeros(len(_c))
    _ = evaluate_interior(_eval_pts, _uf, _qf, _dp, _dw, _no)

    for N in [8, 16, 32]:
        nodes, elems = generate_bumpy_mesh(N)
        centroids, areas, normals, elems = compute_geometry(nodes, elems)
        bc_type, bc_val = assign_mixed_bcs(centroids, normals)

        t0 = time.time()
        dun_pts, dun_wts, near_pts, near_wts = precompute_integrations(nodes, elems, areas, gp_near, gw_near)

        Ne = centroids.shape[0]

        # Disk-backed Matrix initialization
        fd, temp_path = tempfile.mkstemp(suffix='.dat')
        os.close(fd)
        A = np.memmap(temp_path, dtype=np.float64, mode='w+', shape=(Ne, Ne))
        b = np.zeros(Ne, dtype=np.float64)

        assemble_system(centroids, areas, nodes, elems, dun_pts, dun_wts, near_pts, near_wts, gp_self, gw_self, normals, bc_type, bc_val, A, b)
        t_setup = time.time() - t0

        t1 = time.time()

        # Iterative Solve using GMRES and Out-Of-Core streaming
        A_op = OutOfCoreMatrix(A, (Ne, Ne))
        x, info = spla.gmres(A_op, b, rtol=1e-5, restart=100)

        t_solve = time.time() - t1

        u_full = np.zeros(Ne)
        q_full = np.zeros(Ne)

        dir_mask = (bc_type == 0)
        neu_mask = (bc_type == 1)

        u_full[dir_mask] = bc_val[dir_mask]
        q_full[dir_mask] = x[dir_mask]

        u_full[neu_mask] = x[neu_mask]
        q_full[neu_mask] = bc_val[neu_mask]

        xi = np.linspace(-0.5, 0.5, 5)
        yi = np.linspace(-0.5, 0.5, 5)
        zi = np.linspace(-0.5, 0.5, 5)
        XX, YY, ZZ = np.meshgrid(xi, yi, zi)
        eval_pts = np.ascontiguousarray(np.vstack((XX.flatten(), YY.flatten(), ZZ.flatten())).T)

        t2 = time.time()
        u_int_num = evaluate_interior(eval_pts, u_full, q_full, dun_pts, dun_wts, normals)
        t_eval = time.time() - t2

        u_int_exact = np.sinh(eval_pts[:, 0])*np.sin(eval_pts[:, 1]) + np.cosh(eval_pts[:, 1])*np.cos(eval_pts[:, 2])
        rel_l2_error = np.linalg.norm(u_int_num - u_int_exact) / np.linalg.norm(u_int_exact)
        t_total = t_setup + t_solve + t_eval

        print(f"{N:<4} | {Ne:<7} | {rel_l2_error:.6e}   | {t_setup:.4f}    | {t_solve:.4f}    | {t_eval:.4f}   | {t_total:.4f}")
        results.append((N, rel_l2_error))

        del A, A_op
        os.remove(temp_path)
        del b, x, u_full, q_full, nodes, elems, centroids, areas, normals, dun_pts, dun_wts, near_pts, near_wts
        gc.collect()

    slope = -(np.log(results[-1][1]) - np.log(results[0][1])) / (np.log(results[-1][0]) - np.log(results[0][0]))
    print("Convergence Analysis:")
    print(f"Computed Slope: {slope:.4f}")
    print("Expected Slope: ~1.0000 (O(h) for constant elements)")

if __name__ == '__main__':
    main()