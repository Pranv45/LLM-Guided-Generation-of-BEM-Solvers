import numpy as np
import time

def bem_spectral_solve(M, n_mode):
    """
    Spectral BEM on the unit circle using Fourier diagonalization.
    Solves (1/2 I + K) μ = cos(nθ).
    """
    theta = np.linspace(0, 2*np.pi, M, endpoint=False)
    f = np.cos(n_mode * theta)

    # Fourier transform
    fhat = np.fft.fft(f)

    # Eigenvalues of (1/2 I + K) for Laplace double-layer on unit circle
    k = np.fft.fftfreq(M) * M
    lam = np.zeros(M)

    lam[k == 0] = 0.5
    lam[k != 0] = 0.5

    # Solve in Fourier space
    muhat = fhat / lam
    mu = np.real(np.fft.ifft(muhat))

    return theta, mu

def bem_spectral_evaluate(theta, mu, pts):
    """
    Evaluate double-layer potential inside unit disk.
    """
    nodes = np.column_stack([np.cos(theta), np.sin(theta)])
    normals = nodes.copy()
    w = 2*np.pi / len(theta)

    u = np.zeros(len(pts))

    for j in range(len(theta)):
        yj = nodes[j]
        nj = normals[j]
        r = pts - yj
        r2 = np.sum(r*r, axis=1)
        kern = -(1/(2*np.pi)) * (r @ nj) / r2
        u += kern * mu[j] * w

    return u

import pandas as pd
import numpy as np
import time

def run_spectral_bem_study(M_list, n_mode=3):
    records = []

    # fixed evaluation grid (same as before)
    Nx = Ny = 120
    xx = np.linspace(-0.9, 0.9, Nx)
    yy = np.linspace(-0.9, 0.9, Ny)
    X, Y = np.meshgrid(xx, yy)
    mask = (X**2 + Y**2) < 0.9**2
    pts = np.column_stack([X[mask], Y[mask]])

    # exact solution on evaluation grid
    R = np.sqrt(np.sum(pts**2, axis=1))
    Theta = np.arctan2(pts[:,1], pts[:,0])
    u_exact = (R**n_mode) * np.cos(n_mode * Theta)

    for M in M_list:
        print(f"\n--- Spectral BEM | M={M} ---")

        # ---------- Solve (Fourier diagonalization) ----------
        t0 = time.perf_counter()
        theta, mu = bem_spectral_solve(M, n_mode)
        t_solve = time.perf_counter() - t0

        # ---------- Evaluation ----------
        t0 = time.perf_counter()
        u_num = bem_spectral_evaluate(theta, mu, pts)
        t_eval = time.perf_counter() - t0

        # ---------- Error ----------
        relL2 = np.linalg.norm(u_num - u_exact) / np.linalg.norm(u_exact)

        total_time = t_solve + t_eval

        records.append({
            "M": M,
            "Unknowns": M,          # boundary DOFs
            "Iterations": 1,        # diagonal solve
            "t_solve": t_solve,
            "t_eval": t_eval,
            "t_total": total_time,
            "relL2": relL2
        })

        print(
            f"M={M:6d}, "
            f"t_total={total_time:.4e}s, "
            f"relL2={relL2:.2e}"
        )

    return pd.DataFrame(records)

M_list = [100, 200, 400, 800, 1600]
df_bem_spectral = run_spectral_bem_study(M_list, n_mode=3)

print(df_bem_spectral)
