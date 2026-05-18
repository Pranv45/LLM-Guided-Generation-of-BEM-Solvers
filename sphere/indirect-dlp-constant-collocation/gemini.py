import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
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
            # Sort to ensure the edge is direction-independent
            edge = tuple(sorted([v1_idx, v2_idx]))
            if edge in v_dict:
                return v_dict[edge]

            # Get points, average, and project to unit sphere
            pt1 = np.array(new_vertices[v1_idx])
            pt2 = np.array(new_vertices[v2_idx])
            mid = (pt1 + pt2) / 2.0
            mid /= np.linalg.norm(mid)

            # Assign the new index based on the growing list
            new_idx = len(new_vertices)
            v_dict[edge] = new_idx
            new_vertices.append(mid.tolist())
            return new_idx

        for face in faces:
            v0, v1, v2 = face

            # Get midpoints for each edge
            a = get_midpoint(v0, v1)
            b = get_midpoint(v1, v2)
            c = get_midpoint(v2, v0)

            # Create the 4 new subdivided triangles
            new_faces.extend([
                [v0, a, c],
                [v1, b, a],
                [v2, c, b],
                [a, b, c]
            ])

        vertices = np.array(new_vertices)
        faces = np.array(new_faces)

    return vertices, faces

def compute_geometry(vertices, faces):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    centroids = (v0 + v1 + v2) / 3.0
    cross = np.cross(v1 - v0, v2 - v0)
    areas = np.linalg.norm(cross, axis=1) / 2.0
    normals = cross / (2.0 * areas[:, np.newaxis])

    dots = np.sum(normals * centroids, axis=1)
    flip = dots < 0
    normals[flip] *= -1.0

    return centroids, areas, normals

def get_quadrature(vertices, faces, areas):
    bary = np.array([
        [1/3, 1/3, 1/3],
        [0.0597158717, 0.4701420641, 0.4701420641],
        [0.4701420641, 0.0597158717, 0.4701420641],
        [0.4701420641, 0.4701420641, 0.0597158717],
        [0.7974269853, 0.1012865073, 0.1012865073],
        [0.1012865073, 0.7974269853, 0.1012865073],
        [0.1012865073, 0.1012865073, 0.7974269853]
    ])
    weights = np.array([
        0.225,
        0.1323941527, 0.1323941527, 0.1323941527,
        0.1259391805, 0.1259391805, 0.1259391805
    ])

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    Ne = len(faces)
    Nq = len(weights)

    quad_pts = np.zeros((Ne, Nq, 3))
    quad_wts = np.zeros((Ne, Nq))

    for q in range(Nq):
        quad_pts[:, q, :] = bary[q, 0] * v0 + bary[q, 1] * v1 + bary[q, 2] * v2
        quad_wts[:, q] = weights[q] * areas

    return quad_pts, quad_wts

@njit(parallel=True, fastmath=True)
def matvec_bem(mu, centroids, quad_pts, quad_wts, normals):
    Ne = centroids.shape[0]
    Nq = quad_pts.shape[1]
    y = np.empty(Ne, dtype=np.float64)

    for i in prange(Ne):
        cx = centroids[i, 0]
        cy = centroids[i, 1]
        cz = centroids[i, 2]

        val = -0.5 * mu[i]

        for j in range(Ne):
            if i == j:
                continue

            nx = normals[j, 0]
            ny = normals[j, 1]
            nz = normals[j, 2]
            mu_j = mu[j]

            for q in range(Nq):
                dx = cx - quad_pts[j, q, 0]
                dy = cy - quad_pts[j, q, 1]
                dz = cz - quad_pts[j, q, 2]

                dist3 = (dx*dx + dy*dy + dz*dz)**1.5
                dot = dx*nx + dy*ny + dz*nz

                val += (1.0 / (4.0 * np.pi)) * (dot / dist3) * quad_wts[j, q] * mu_j

        y[i] = val
    return y

@njit(parallel=True, fastmath=True)
def eval_interior(grid_pts, mu, quad_pts, quad_wts, normals):
    Np = grid_pts.shape[0]
    Ne = quad_pts.shape[0]
    Nq = quad_pts.shape[1]
    u = np.zeros(Np, dtype=np.float64)

    for i in prange(Np):
        px = grid_pts[i, 0]
        py = grid_pts[i, 1]
        pz = grid_pts[i, 2]

        val = 0.0
        for j in range(Ne):
            nx = normals[j, 0]
            ny = normals[j, 1]
            nz = normals[j, 2]
            mu_j = mu[j]

            for q in range(Nq):
                dx = px - quad_pts[j, q, 0]
                dy = py - quad_pts[j, q, 1]
                dz = pz - quad_pts[j, q, 2]

                dist3 = (dx*dx + dy*dy + dz*dz)**1.5
                dot = dx*nx + dy*ny + dz*nz

                val += (1.0 / (4.0 * np.pi)) * (dot / dist3) * quad_wts[j, q] * mu_j

        u[i] = val
    return u

def exact_solution(pts):
    return pts[:, 0]**2 - pts[:, 1]**2

def main():
    x = np.linspace(-0.8, 0.8, 20)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    r = np.linalg.norm(pts, axis=1)
    grid_pts = pts[r < 0.8]
    u_exact_interior = exact_solution(grid_pts)

    levels = [3, 4, 5]
    results = []

    print("-" * 85)
    print(f"{'Level':<6} | {'Ne':<6} | {'Iters':<6} | {'L2 Error':<12} | {'Setup(s)':<8} | {'Solve(s)':<8} | {'Eval(s)':<8} | {'Total(s)':<8}")
    print("-" * 85)

    for level in levels:
        t_start = time.time()

        base_v, base_f = create_icosahedron()
        vertices, faces = subdivide(base_v, base_f, level)
        centroids, areas, normals = compute_geometry(vertices, faces)
        quad_pts, quad_wts = get_quadrature(vertices, faces, areas)

        Ne = len(faces)
        u_exact_bc = exact_solution(centroids)

        t_setup = time.time() - t_start

        t_solve_start = time.time()
        counter = [0]
        def callback(pr_norm):
            counter[0] += 1

        A = LinearOperator((Ne, Ne), matvec=lambda x: matvec_bem(x, centroids, quad_pts, quad_wts, normals))
        mu, info = gmres(A, u_exact_bc, rtol=1e-8, atol=1e-8, restart=Ne, callback=callback, callback_type='pr_norm')

        t_solve = time.time() - t_solve_start

        t_eval_start = time.time()
        u_num = eval_interior(grid_pts, mu, quad_pts, quad_wts, normals)
        t_eval = time.time() - t_eval_start

        t_total = time.time() - t_start

        l2_err = np.linalg.norm(u_num - u_exact_interior) / np.linalg.norm(u_exact_interior)
        iters = counter[0]

        results.append({
            'Ne': Ne,
            'h': np.sqrt(1.0 / Ne),
            'err': l2_err
        })

        print(f"{level:<6} | {Ne:<6} | {iters:<6} | {l2_err:<12.5e} | {t_setup:<8.3f} | {t_solve:<8.3f} | {t_eval:<8.3f} | {t_total:<8.3f}")

    print("-" * 85)

    log_h = np.log([r['h'] for r in results])
    log_e = np.log([r['err'] for r in results])
    A_fit = np.vstack([log_h, np.ones(len(log_h))]).T
    m, c = np.linalg.lstsq(A_fit, log_e, rcond=None)[0]

    print(f"Estimated convergence order (slope): {m:.4f}")

if __name__ == "__main__":
    main()