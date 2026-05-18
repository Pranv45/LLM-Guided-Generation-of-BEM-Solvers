import numpy as np
from scipy.sparse.linalg import gmres
from numba import njit, prange
import time

def create_icosahedron():
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array([
        [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1]
    ], dtype=np.float64)
    vertices /= np.linalg.norm(vertices[0])

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=np.int32)
    return vertices, faces

def subdivide(vertices, faces, levels):
    for _ in range(levels):
        v_dict = {}
        new_faces = []
        new_vertices = vertices.tolist()

        def get_midpoint(v1_idx, v2_idx):
            edge = tuple(sorted([v1_idx, v2_idx]))
            if edge in v_dict:
                return v_dict[edge]
            pt1 = np.array(new_vertices[v1_idx])
            pt2 = np.array(new_vertices[v2_idx])
            mid = (pt1 + pt2) / 2.0
            mid /= np.linalg.norm(mid)
            new_idx = len(new_vertices)
            v_dict[edge] = new_idx
            new_vertices.append(mid.tolist())
            return new_idx

        for face in faces:
            v0, v1, v2 = face
            a = get_midpoint(v0, v1)
            b = get_midpoint(v1, v2)
            c = get_midpoint(v2, v0)

            new_faces.extend([
                [v0, a, c], [v1, b, a], [v2, c, b], [a, b, c]
            ])

        vertices = np.array(new_vertices)
        faces = np.array(new_faces)
    return vertices, faces

def compute_geometry(vertices, faces):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    cross = np.cross(v1 - v0, v2 - v0)
    areas = np.linalg.norm(cross, axis=1) / 2.0
    normals = cross / (2.0 * areas[:, np.newaxis])

    centroids = (v0 + v1 + v2) / 3.0
    dots = np.sum(normals * centroids, axis=1)
    flip = dots < 0
    normals[flip] *= -1.0

    return areas, normals

def get_composite_quadrature(L):
    bary_7 = np.array([
        [1/3, 1/3, 1/3],
        [0.0597158717, 0.4701420641, 0.4701420641],
        [0.4701420641, 0.0597158717, 0.4701420641],
        [0.4701420641, 0.4701420641, 0.0597158717],
        [0.7974269853, 0.1012865073, 0.1012865073],
        [0.1012865073, 0.7974269853, 0.1012865073],
        [0.1012865073, 0.1012865073, 0.7974269853]
    ])
    weights_7 = np.array([
        0.225, 0.1323941527, 0.1323941527, 0.1323941527,
        0.1259391805, 0.1259391805, 0.1259391805
    ])

    tris = [np.array([[1.0,0,0], [0,1.0,0], [0,0,1.0]])]
    for _ in range(L):
        new_tris = []
        for t in tris:
            m0 = (t[0] + t[1]) / 2.0
            m1 = (t[1] + t[2]) / 2.0
            m2 = (t[2] + t[0]) / 2.0
            new_tris.extend([
                np.array([t[0], m0, m2]), np.array([t[1], m1, m0]),
                np.array([t[2], m2, m1]), np.array([m0, m1, m2])
            ])
        tris = new_tris

    final_bary = []
    final_wts = []
    area_factor = 1.0 / (4**L)

    for t in tris:
        for i in range(len(weights_7)):
            pt = bary_7[i,0]*t[0] + bary_7[i,1]*t[1] + bary_7[i,2]*t[2]
            final_bary.append(pt)
            final_wts.append(weights_7[i] * area_factor)

    return np.array(final_bary, dtype=np.float64), np.array(final_wts, dtype=np.float64)

@njit(parallel=True, fastmath=True)
def assemble_A_b(A, b, nodes, faces, areas, normals, offsets, node2elems, q_bary_0, q_wts_0, q_bary_2, q_wts_2):
    N = nodes.shape[0]
    Ne = faces.shape[0]

    for i in prange(N):
        row = np.zeros(N, dtype=np.float64)
        b_val = 0.0

        start = offsets[i]
        end = offsets[i+1]

        for idx in range(start, end):
            e_x = node2elems[idx]
            v_x = faces[e_x]

            if v_x[0] == i: k_x = 0
            elif v_x[1] == i: k_x = 1
            else: k_x = 2

            area_x = areas[e_x]
            for local_j in range(3):
                j = v_x[local_j]
                if k_x == local_j:
                    row[j] += -0.5 * (area_x / 6.0)
                else:
                    row[j] += -0.5 * (area_x / 12.0)

            for q in range(len(q_wts_0)):
                qx = q_bary_0[q,0]*nodes[v_x[0],0] + q_bary_0[q,1]*nodes[v_x[1],0] + q_bary_0[q,2]*nodes[v_x[2],0]
                qy = q_bary_0[q,0]*nodes[v_x[0],1] + q_bary_0[q,1]*nodes[v_x[1],1] + q_bary_0[q,2]*nodes[v_x[2],1]
                qz = q_bary_0[q,0]*nodes[v_x[0],2] + q_bary_0[q,1]*nodes[v_x[1],2] + q_bary_0[q,2]*nodes[v_x[2],2]

                f_val = qx*qx - qy*qy
                b_val += q_wts_0[q] * area_x * q_bary_0[q, k_x] * f_val

            for e_y in range(Ne):
                if e_x == e_y: continue

                v_y = faces[e_y]
                is_adj = (
                    v_x[0] == v_y[0] or v_x[0] == v_y[1] or v_x[0] == v_y[2] or
                    v_x[1] == v_y[0] or v_x[1] == v_y[1] or v_x[1] == v_y[2] or
                    v_x[2] == v_y[0] or v_x[2] == v_y[1] or v_x[2] == v_y[2]
                )

                if is_adj:
                    qb_x, qw_x = q_bary_2, q_wts_2
                    qb_y, qw_y = q_bary_2, q_wts_2
                else:
                    qb_x, qw_x = q_bary_0, q_wts_0
                    qb_y, qw_y = q_bary_0, q_wts_0

                nx = normals[e_y, 0]
                ny = normals[e_y, 1]
                nz = normals[e_y, 2]
                area_factor = area_x * areas[e_y]

                val0 = 0.0
                val1 = 0.0
                val2 = 0.0

                for qx in range(len(qw_x)):
                    px = qb_x[qx,0]*nodes[v_x[0],0] + qb_x[qx,1]*nodes[v_x[1],0] + qb_x[qx,2]*nodes[v_x[2],0]
                    py = qb_x[qx,0]*nodes[v_x[0],1] + qb_x[qx,1]*nodes[v_x[1],1] + qb_x[qx,2]*nodes[v_x[2],1]
                    pz = qb_x[qx,0]*nodes[v_x[0],2] + qb_x[qx,1]*nodes[v_x[1],2] + qb_x[qx,2]*nodes[v_x[2],2]

                    w_x_phi = qw_x[qx] * qb_x[qx, k_x]

                    for qy in range(len(qw_y)):
                        sy = qb_y[qy,0]*nodes[v_y[0],0] + qb_y[qy,1]*nodes[v_y[1],0] + qb_y[qy,2]*nodes[v_y[2],0]
                        ty = qb_y[qy,0]*nodes[v_y[0],1] + qb_y[qy,1]*nodes[v_y[1],1] + qb_y[qy,2]*nodes[v_y[2],1]
                        uz = qb_y[qy,0]*nodes[v_y[0],2] + qb_y[qy,1]*nodes[v_y[1],2] + qb_y[qy,2]*nodes[v_y[2],2]

                        dx = px - sy
                        dy = py - ty
                        dz = pz - uz

                        dist3 = (dx*dx + dy*dy + dz*dz)**1.5
                        dot = dx*nx + dy*ny + dz*nz

                        kernel = (1.0 / (4.0 * np.pi)) * (dot / dist3) * w_x_phi * qw_y[qy]

                        val0 += kernel * qb_y[qy, 0]
                        val1 += kernel * qb_y[qy, 1]
                        val2 += kernel * qb_y[qy, 2]

                row[v_y[0]] += val0 * area_factor
                row[v_y[1]] += val1 * area_factor
                row[v_y[2]] += val2 * area_factor

        A[i, :] = row
        b[i] = b_val

@njit(parallel=True, fastmath=True)
def eval_interior(grid_pts, mu, nodes, faces, areas, normals, q_bary, q_wts):
    Np = grid_pts.shape[0]
    Ne = faces.shape[0]
    u = np.zeros(Np, dtype=np.float64)

    for i in prange(Np):
        px = grid_pts[i, 0]
        py = grid_pts[i, 1]
        pz = grid_pts[i, 2]
        val = 0.0

        for e in range(Ne):
            v = faces[e]
            nx = normals[e, 0]
            ny = normals[e, 1]
            nz = normals[e, 2]

            mu0 = mu[v[0]]
            mu1 = mu[v[1]]
            mu2 = mu[v[2]]

            for q in range(len(q_wts)):
                sx = q_bary[q,0]*nodes[v[0],0] + q_bary[q,1]*nodes[v[1],0] + q_bary[q,2]*nodes[v[2],0]
                sy = q_bary[q,0]*nodes[v[0],1] + q_bary[q,1]*nodes[v[1],1] + q_bary[q,2]*nodes[v[2],1]
                sz = q_bary[q,0]*nodes[v[0],2] + q_bary[q,1]*nodes[v[1],2] + q_bary[q,2]*nodes[v[2],2]

                dx = px - sx
                dy = py - sy
                dz = pz - sz

                dist3 = (dx*dx + dy*dy + dz*dz)**1.5
                dot = dx*nx + dy*ny + dz*nz

                kernel = (1.0 / (4.0 * np.pi)) * (dot / dist3) * q_wts[q] * areas[e]
                mu_q = q_bary[q, 0]*mu0 + q_bary[q, 1]*mu1 + q_bary[q, 2]*mu2

                val += kernel * mu_q
        u[i] = val
    return u

def main():
    x = np.linspace(-0.8, 0.8, 30)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    r = np.linalg.norm(pts, axis=1)
    grid_pts = pts[(r > 0.1) & (r < 0.8)]
    u_exact_interior = grid_pts[:, 0]**2 - grid_pts[:, 1]**2

    q_bary_0, q_wts_0 = get_composite_quadrature(0)
    q_bary_2, q_wts_2 = get_composite_quadrature(2)

    levels = [3, 4, 5]
    results = []

    print("-" * 88)
    print(f"{'Level':<6} | {'N':<6} | {'Iters':<6} | {'L2 Error':<12} | {'Assemble':<9} | {'Solve':<9} | {'Eval':<9} | {'Total':<9}")
    print("-" * 88)

    for level in levels:
        t_start = time.time()

        base_v, base_f = create_icosahedron()
        nodes, faces = subdivide(base_v, base_f, level)
        areas, normals = compute_geometry(nodes, faces)

        N = len(nodes)

        counts = np.bincount(faces.ravel(), minlength=N)
        offsets = np.zeros(N + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(counts)
        node2elems = np.zeros(offsets[-1], dtype=np.int32)
        current_offsets = offsets[:-1].copy()

        for e in range(len(faces)):
            for v in faces[e]:
                node2elems[current_offsets[v]] = e
                current_offsets[v] += 1

        A = np.zeros((N, N), dtype=np.float64)
        b = np.zeros(N, dtype=np.float64)

        t_asm_start = time.time()
        assemble_A_b(A, b, nodes, faces, areas, normals, offsets, node2elems, q_bary_0, q_wts_0, q_bary_2, q_wts_2)
        t_asm = time.time() - t_asm_start

        t_solve_start = time.time()
        counter = [0]
        def callback(pr_norm):
            counter[0] += 1

        mu, _ = gmres(A, b, rtol=1e-8, atol=1e-8, restart=N, callback=callback, callback_type='pr_norm')
        t_solve = time.time() - t_solve_start

        t_eval_start = time.time()
        u_num = eval_interior(grid_pts, mu, nodes, faces, areas, normals, q_bary_0, q_wts_0)
        t_eval = time.time() - t_eval_start

        t_total = time.time() - t_start

        l2_err = np.linalg.norm(u_num - u_exact_interior) / np.linalg.norm(u_exact_interior)

        results.append({
            'h': np.sqrt(1.0 / N),
            'err': l2_err
        })

        print(f"{level:<6} | {N:<6} | {counter[0]:<6} | {l2_err:<12.5e} | {t_asm:<9.3f} | {t_solve:<9.3f} | {t_eval:<9.3f} | {t_total:<9.3f}")

    print("-" * 88)

    log_h = np.log([r['h'] for r in results])
    log_e = np.log([r['err'] for r in results])
    A_fit = np.vstack([log_h, np.ones(len(log_h))]).T
    m, c = np.linalg.lstsq(A_fit, log_e, rcond=None)[0]

    print(f"Estimated convergence order (slope): {m:.4f}")

if __name__ == "__main__":
    main()