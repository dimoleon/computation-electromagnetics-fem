import numpy as np 
import meshes
import scipy.sparse as ssp
import scipy.sparse.linalg as ssl
import matplotlib.pyplot as plt 
import matplotlib.tri as tri

w = 4e-2
t = 0.2e-2 
d = 1e-2
V = 100
er = 2.2 
domain_factor = 3
refinements = 2

UPPERPOTENTIAL = V/2 
LOWERPOTENTIAL = -V/2 

msh = meshes.parallel_plates_mesh(w, t, d, domain_factor, refinements)

points = msh.points[:, :2]
edges = msh.cells_dict["line"]
triangles = msh.cells_dict["triangle"]

upper_lines_idxs = msh.cell_sets_dict["upper_plate"]["line"]
lower_lines_idxs = msh.cell_sets_dict["lower_plate"]["line"]

air_triangles_idxs = msh.cell_sets_dict["air_surface"]["triangle"]
dielectric_triangles_idxs = msh.cell_sets_dict["dielectric_surface"]["triangle"]

upper_nodes = set() 
lower_nodes = set() 

for e in edges[upper_lines_idxs]: 
    upper_nodes.update(e) 

for e in edges[lower_lines_idxs]: 
    lower_nodes.update(e) 

dirichlet_nodes = upper_nodes.union(lower_nodes)

num_points = len(points)
num_edges = len(edges) 
num_trian = len(triangles) 

X0 = np.zeros(num_points)
X0[list(upper_nodes)] = UPPERPOTENTIAL
X0[list(lower_nodes)] = LOWERPOTENTIAL

known = np.array(list(dirichlet_nodes))
unknown = [] 
for n in np.arange(num_points): 
    if n not in dirichlet_nodes: 
        unknown.append(n) 

index = np.full(num_points, -1) 
index[unknown] = [i for i in range(len(unknown))]

node_id = np.full(num_points, False) 
node_id[known] = True

Nf = len(unknown) 
S = ssp.lil_matrix((Nf, Nf))

B = np.zeros(Nf) 
grad = np.empty(num_trian, dtype=object)

def epsilon(idt):
    e = 1
    if idt in dielectric_triangles_idxs:
        e = er
    return e

for idt, t in enumerate(triangles): 
    x = [p[0] for p in points[t]]
    y = [p[1] for p in points[t]]
    mat = np.array([[1, x[0], y[0]], 
                    [1, x[1], y[1]],
                    [1, x[2], y[2]]])
    De = np.linalg.det(mat)
    area = abs(De/2)
    
    b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])/De
    c = -np.array([x[1]-x[2], x[2]-x[0], x[0]-x[1]])/De
    grad[idt] = np.array([b, c])

    Se = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            Se[i, j] = (b[i]*b[j] + c[i]*c[j])*area*epsilon(idt)
            if not node_id[t[i]]: 
                if not node_id[t[j]]: 
                    S[index[t[i]], index[t[j]]] += Se[i, j]
                else:
                    B[index[t[i]]] -= Se[i, j]*X0[t[j]]


X = np.array(ssl.spsolve(S, B))
X0[unknown] = X

triang = tri.Triangulation(points[:, 0], points[:, 1], triangles)

E_per_element = np.array([-grad[idt] @ X0[tri] for idt, tri in enumerate(triangles)])

E_at_nodes = np.zeros((len(points), 2))
count = np.zeros(len(points)) 

for tri, E in zip(triangles, E_per_element): 
    E_at_nodes[tri] += E
    count[tri] += 1 

for i in range(num_points): 
    E_at_nodes[i] /= count[i]

# plt.tripcolor(triang, np.linalg.norm(E_at_nodes, axis=1), norm='log', cmap='viridis')
plt.figure()
plt.tricontour(triang, X0, colors='black')
plt.tricontourf(triang, X0, levels=50, cmap='RdBu_r')
plt.colorbar(label="Potential")
plt.title("Electrostatic Potential")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.tight_layout()
plt.show()

plt.figure()
plt.quiver(points[:, 0], points[:, 1], E_at_nodes[:, 0], E_at_nodes[:, 1], scale=5*np.max(np.linalg.norm(E_at_nodes)), cmap='plasma')
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.tight_layout()
plt.show()
