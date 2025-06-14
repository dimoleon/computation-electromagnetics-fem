import numpy as np 
import meshes 
import scipy.sparse as ssp
import scipy.sparse.linalg as ssl
import matplotlib.pyplot as plt
import matplotlib.tri as tri

a = 0.01
refinements = 3
num_eigs = 6

msh = meshes.circular_waveguide_mesh(a, refinements=refinements)

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

def local_stiffness_mass(p1, p2, p3): 
    # x = [p[0] for p in [p1, p2, p3]]
    # y = [p[1] for p in [p1, p2, p3]]
    # mat = np.array([[1, x[0], y[0]], 
    #                 [1, x[1], y[1]],
    #                 [1, x[2], y[2]]])
    # De = np.linalg.det(mat)
    # area = abs(De/2)
    
    # b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])/De
    # c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])/De
    # grad[idt] = np.array([b, c])
    
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    
    v1 = p2 - p1  
    v2 = p3 - p1  
    
    area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
    
    grad_N = np.array([
        [p2[1] - p3[1], p3[0] - p2[0]],  
        [p3[1] - p1[1], p1[0] - p3[0]], 
        [p1[1] - p2[1], p2[0] - p1[0]]  
    ]) / (2 * area)
    
    Se = area * (grad_N @ grad_N.T)
    
    # Mass matrix
    Te = (area/12) * np.array([[2, 1, 1],
                               [1, 2, 1], 
                               [1, 1, 2]])
    
    return Se, Te    
    # Se = np.zeros((3,3))
    # for i in range(3):
    #     for j in range(3):
    #         Se[i, j] = (b[i]*b[j] + c[i]*c[j])*area

    # Te = (area/12) * (np.ones((3, 3)) + np.eye(3))
    # return Se, Te

S = ssp.lil_matrix((num_points, num_points))
T = ssp.lil_matrix((num_points, num_points))

for t in triangles: 
    verts = points[t]
    Se, Te = local_stiffness_mass(verts[0], verts[1], verts[2])

    for i in range(3): 
        for j in range(3): 
            S[t[i], t[j]] += Se[i, j]
            T[t[i], t[j]] += Te[i, j]

S = ssp.csr_matrix(S)
T = ssp.csr_matrix(T)
eigenvalues, eigenvectors = ssl.eigsh(S, num_eigs+1, M=T, sigma=0.1)
idx = np.argsort(eigenvalues)
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

nonzero = eigenvalues > 1e-6
eigenvalues = eigenvalues[nonzero][:num_eigs]
eigenvectors = eigenvectors[:, nonzero][:, :num_eigs]

# def cutoff_frequencies(eig): 
c = 3e8
# cutoff_freqs = np.sqrt(eigenvalues) * c / (2 * a * np.pi)
print(np.sqrt(eigenvalues))

# fig, axes = plt.subplots(1, 6)
triang = tri.Triangulation(points[:, 0], points[:, 1], triangles)
theta = np.linspace(0, 2*np.pi, 100)
circle_x = a * np.cos(theta)
circle_y = a * np.sin(theta)
for i in range(2): 
    plt.figure()
    Hz = eigenvectors[:, i]

    # levels = np.linspace(np.min(Hz), np.max(Hz), 20)
    plt.tricontour(triang, Hz, levels=10, colors='black', linewidths=0.3)
    plt.tricontourf(triang, Hz, levels=50)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.colorbar(label="Hz")

    plt.tight_layout() 
    plt.show() 
