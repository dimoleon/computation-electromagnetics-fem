import gmsh
import meshio
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

# Parameters
a = 0.02  # radius [m]
num_eigs = 6
c = 299792458  # speed of light [m/s]

# --- GMSH: Create Mesh ---
gmsh.initialize()
gmsh.model.add("circle")
gmsh.model.occ.addDisk(0, 0, 0, a, a)
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
gmsh.model.mesh.refine()
gmsh.model.mesh.refine() 
gmsh.write("circle.msh")
gmsh.finalize()

# --- meshio: Read Mesh ---
msh = meshio.read("circle.msh")
points = msh.points[:, :2]
cells = msh.cells_dict["triangle"]

# --- Assemble FEM Matrices ---
n_pts = len(points)
K = lil_matrix((n_pts, n_pts))
M = lil_matrix((n_pts, n_pts))

def local_matrices(p1, p2, p3):
    x = np.array([p1, p2, p3])
    area = 0.5 * np.linalg.det(np.array([[1, *p1], [1, *p2], [1, *p3]]))
    b = np.array([p2[1] - p3[1], p3[1] - p1[1], p1[1] - p2[1]])
    c = np.array([p3[0] - p2[0], p1[0] - p3[0], p2[0] - p1[0]])
    B = np.vstack([b, c]) / (2 * area)
    Ke = (B.T @ B) * area
    Me = (area / 12) * (np.ones((3, 3)) + np.eye(3))
    return Ke, Me

for tri in cells:
    idx = tri
    p = points[idx]
    Ke, Me = local_matrices(*p)
    for i in range(3):
        for j in range(3):
            K[idx[i], idx[j]] += Ke[i, j]
            M[idx[i], idx[j]] += Me[i, j]

K = csr_matrix(K)
M = csr_matrix(M)

# --- Solve Generalized Eigenvalue Problem ---
eigs, vecs = eigsh(K, k=num_eigs, M=M, sigma=0, which="LM")
# cutoff_freqs = np.sqrt(eigs) * c / (2 * np.pi)

print(np.sqrt(eigs))