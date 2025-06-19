import numpy as np 
import meshes 
import utils
import scipy.sparse as ssp
import scipy.sparse.linalg as ssl
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.constants import c

a = 0.01
refinements = 3
num_eigs = 6

bessel_root = [2.404825558, 3.831705970, 5.135622302, 5.520078110, 6.380161896, 7.015586669]
n_tm = [0, 1, 2, 0, 3, 1]
m_tm = [1, 1, 1, 2, 1, 2]

bes_der_root = [1.841183781, 3.054236928, 3.831705970, 4.201188941, 5.318288601 , 5.331441994]
n_te = [1, 2, 0, 3, 4, 1]
m_te = [1, 1, 1, 1, 1, 2]

def b1(ref=refinements): 
    msh = meshes.circular_waveguide_mesh(a, refinements=ref)

    points = msh.points[:, :2]
    edges = msh.cells_dict["line"]
    triangles = msh.cells_dict["triangle"]

    boundary_lines_idxs = msh.cell_sets_dict["boundary"]["line"]

    boundary_nodes = set() 

    for e in edges[boundary_lines_idxs]: 
        boundary_nodes.update(e) 

    num_points = len(points)
    num_edges = len(edges)
    num_trian = len(triangles) 
    print(f"Num points: {num_points}")
    print(f"Num triangles: {num_trian}")

    S = ssp.lil_matrix((num_points, num_points))
    T = ssp.lil_matrix((num_points, num_points))

    for t in triangles: 
        Se, Te, _, _ = utils.local_stiffness_mass_grad_area(points, *t)

        for i in range(3): 
            for j in range(3): 
                S[t[i], t[j]] += Se[i, j]
                T[t[i], t[j]] += Te[i, j]

    S = ssp.csr_matrix(S)
    T = ssp.csr_matrix(T)

    eigenvalues, eigenvectors = ssl.eigsh(S, 2*num_eigs, M=T, sigma=0.1)
    _, idxs = np.unique(np.round(eigenvalues, decimals=-2), return_index=True)
    eigenvalues = eigenvalues[idxs]
    eigenvectors = eigenvectors[:, idxs]

    nonzero = eigenvalues > 1e-6
    eigenvalues = eigenvalues[nonzero][:num_eigs]
    eigenvectors = eigenvectors[:, nonzero][:, :num_eigs]
    cutoff_freqs_TE = np.sqrt(eigenvalues) * c / (2 * np.pi)

    dirichlet_nodes = np.array(list(boundary_nodes))
    free_nodes = np.setdiff1d(np.arange(num_points), dirichlet_nodes)
    S_free = S[free_nodes][:, free_nodes]
    T_free = T[free_nodes][:, free_nodes]

    eigenvalues_f, eigenvectors_f = ssl.eigsh(S_free, 2*num_eigs, M=T_free, sigma=0.1)
    _, idxs_f = np.unique(np.round(eigenvalues_f, decimals=-2), return_index=True)
    eigenvalues_f = eigenvalues_f[idxs_f]
    eigenvectors_f = eigenvectors_f[:, idxs_f]

    cutoff_freqs_TM = np.sqrt(eigenvalues_f) * c / (2 * np.pi)

    rel_errs_TE = []
    rel_errs_TM = []

    print("TE modes")
    print("mode\t\t freq_theor\t\t freq_calc \t rel_err")
    for n, m, root, freq_calc in zip(n_te, m_te, bes_der_root, cutoff_freqs_TE):
        freq_theor = c*root/(2*np.pi*a)
        rel_err = utils.calc_relative_error(freq_theor, freq_calc)
        rel_errs_TE.append(rel_err)
        print(f"{n}{m}\t {freq_theor}\t {freq_calc}\t {rel_err:4f} %\t")

    print("\nTM modes")
    print("mode\t\t freq_theor\t\t freq_calc \t rel_err")
    for n, m, root, freq_calc in zip(n_tm, m_tm, bessel_root, cutoff_freqs_TM):
        freq_theor = c*root/(2*np.pi*a)
        rel_err = utils.calc_relative_error(freq_theor, freq_calc)
        rel_errs_TM.append(rel_err)
        print(f"{n}{m}\t {freq_theor}\t {freq_calc}\t {rel_err:4f} %\t")

    if ref == refinements:
        for i in range(num_eigs): 
            Hz = eigenvectors[:, i]
            utils.plot_scalar(points, triangles, Hz, title=f"TE_{n_te[i]}{m_te[i]} - Hz", color_title="Hz (A/m)")

        for i in range(num_eigs): 
            Ez = np.zeros(num_points)
            Ez[free_nodes] = eigenvectors_f[:, i]
            utils.plot_scalar(points, triangles, Ez, title=f"TM_{n_tm[i]}{m_tm[i]} - Ez", color_title="Ez (V/m)")

    return rel_errs_TE, rel_errs_TM

rel_err_matrix_TE = []
rel_err_matrix_TM = []
for r in range(refinements+1): 
    ref_plurar = 'refinement' if r == 1 else 'refinements'
    print(f"\nProblem B1 with {r} " + ref_plurar)

    te_err, tm_err = b1(r)
    rel_err_matrix_TE.append(te_err)
    rel_err_matrix_TM.append(tm_err)

print("\nTE modes - refinements - relative error")
print("nm\\refinements", end="\t\t")
for j in range(refinements+1): 
    print(j, end="\t\t")
print() 
for i in range(num_eigs): 
    print(f"{n_te[i]}{m_te[i]}", end="\t")
    for j in range(refinements+1): 
        print(f"{rel_err_matrix_TE[j][i]} %", end="\t")
    print()

print("\nTM modes - refinements - relative error")
print("nm\\refinements", end="\t\t")
for j in range(refinements+1): 
    print(j, end="\t\t")
print() 
for i in range(num_eigs): 
    print(f"{n_tm[i]}{m_tm[i]}", end="\t")
    for j in range(refinements+1): 
        print(f"{rel_err_matrix_TM[j][i]} %", end="\t")
    print()
