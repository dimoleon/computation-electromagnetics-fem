import numpy as np 
import meshes 
import scipy.sparse as ssp
import scipy.sparse.linalg as ssl
import matplotlib.pyplot as plt
import matplotlib.tri as tri

E0 = 1
f = 3e8
c = 3e8
omega = 2*np.pi*f
wavelength = c/f
k0 = omega/c
a_sel = [wavelength/4, wavelength, 2.5*wavelength]
diffRa_sel = [wavelength/2, wavelength, 2*wavelength]
refinements = 3


a = a_sel[1]
diffRa = diffRa_sel[2]
R = a + diffRa

msh = meshes.coaxial_mesh(a=a, b=R, refinements=refinements, model_name='cylinder_scatter')

points = msh.points[:, :2]
edges = msh.cells_dict["line"]
triangles = msh.cells_dict["triangle"]

inner_nodes = set() 
outer_nodes = set() 

inner_lines_idxs = msh.cell_sets_dict["inner_boundary"]["line"]
outer_lines_idxs = msh.cell_sets_dict["outer_boundary"]["line"]

def incident_electric_field_value(id): 
    x = points[id][0]
    return E0 * np.exp(-1j * k0 * x) 

def local_stiffness_mass(n1, n2, n3): 
    p1, p2, p3 = points[[n1, n2, n3]]
    
    v1 = p2 - p1  
    v2 = p3 - p1  
    
    area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
    
    grad_N = np.array([
        [p2[1] - p3[1], p3[0] - p2[0]],  
        [p3[1] - p1[1], p1[0] - p3[0]], 
        [p1[1] - p2[1], p2[0] - p1[0]]  
    ]) / (2 * area)
    
    Se = area * (grad_N @ grad_N.T)
    
    Te = (area/12) * np.array([[2, 1, 1],
                               [1, 2, 1], 
                               [1, 1, 2]])
    
    return Se, Te    

for e in edges[inner_lines_idxs]: 
    inner_nodes.update(e)

for e in edges[outer_lines_idxs]: 
    outer_nodes.update(e)

num_points = len(points)
num_edges = len(edges)
num_trian = len(triangles)
ids = np.arange(num_points)

dirichlet_nodes = np.array(list(inner_nodes))
E_inc = np.vectorize(incident_electric_field_value)(ids)
E_scat = np.zeros(num_points, dtype=complex)
E_scat[dirichlet_nodes] = -E_inc[dirichlet_nodes]

known = np.array(dirichlet_nodes)
unknown = np.setdiff1d(np.arange(num_points), known)

index = np.full(num_points, -1) 
index[unknown] = np.arange(len(unknown))

node_id = np.full(num_points, False) 
node_id[known] = True

A = ssp.lil_matrix((num_points, num_points), dtype=complex)
S = ssp.lil_matrix((num_points, num_points), dtype=complex)
T = ssp.lil_matrix((num_points, num_points), dtype=complex)
b = np.zeros(num_points, dtype=complex)

for t in triangles: 
    Se, Te = local_stiffness_mass(*t)
    for i in range(3): 
        for j in range(3): 
                S[t[i], t[j]] += Se[i, j]
                T[t[i], t[j]] += Te[i, j]

A = S - k0**2 * T

for e in edges[outer_lines_idxs]: 
    id1, id2 = e[0], e[1]
    length = np.linalg.norm(points[id1]-points[id2])

    a = (-1j * k0 - 1/(2*R)) 
    mass_mat = (length/6) * np.array([[2, 1], 
                                      [1, 2]])
    ABCe = a * mass_mat
    A[id1, id1] -= ABCe[0, 0]
    A[id1, id2] -= ABCe[0, 1]
    A[id2, id1] -= ABCe[1, 0]
    A[id2, id2] -= ABCe[1, 1]

Aff = A[unknown, :][:, unknown]
Afp = A[unknown, :][:, known]
b = -Afp @ E_scat[known]

E_scat_uknown = np.array(ssl.spsolve(Aff, b))

E_scat[unknown] = E_scat_uknown

E_total = E_scat + E_inc

triang = tri.Triangulation(points[:, 0], points[:, 1], triangles)

plt.figure() 
plt.tripcolor(triang, np.real(E_total), cmap='RdBu_r')
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.tight_layout() 
plt.show() 