import time
import numpy as np
import meshes 
import scipy.sparse as ssp
import scipy.sparse.linalg as ssl
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.constants import epsilon_0


Z = 50
er = 1
epsilon = er*epsilon_0
b = 0.0175
a = b*np.exp(-Z*np.sqrt(er)/60)
refinements = 2
INNERPOTENTIAL = 1.0
OUTERPOTENTIAL = 0.0

msh = meshes.coaxial_mesh(a, b, refinements=refinements)

points = msh.points[:, :2]
edges = msh.cells_dict["line"]
triangles = msh.cells_dict["triangle"]

inner_nodes = set() 
outer_nodes = set() 

inner_lines_idxs = msh.cell_sets_dict["inner_boundary"]["line"]
outer_lines_idxs = msh.cell_sets_dict["outer_boundary"]["line"]

for e in edges[inner_lines_idxs]: 
    inner_nodes.update(e)

for e in edges[outer_lines_idxs]: 
    outer_nodes.update(e)

dirichlet_nodes = inner_nodes.union(outer_nodes)

num_points = len(points)
num_edges = len(edges)
num_trian = len(triangles)

X0 = np.zeros(num_points)
X0[list(inner_nodes)] = INNERPOTENTIAL
X0[list(outer_nodes)] = OUTERPOTENTIAL

known = np.array(list(dirichlet_nodes))
unknown = np.setdiff1d(np.arange(num_points), known) 
# for n in np.arange(num_points): 
#     if n not in dirichlet_nodes:
#         unknown.append(n)

index = np.full(num_points, -1)
index[unknown] = np.arange(len(unknown))

node_id = np.full(num_points, False)
node_id[known] = True

Nf = len(unknown)
S = ssp.lil_matrix((Nf, Nf))
B = np.zeros(Nf)
grad = np.empty(num_trian, dtype=object)
areas = np.zeros(num_trian)

for idt, t in enumerate(triangles): 
    x = points[t][:, 0]
    y = points[t][:, 1]
    mat = np.array([[1, x[0], y[0]], 
                    [1, x[1], y[1]],
                    [1, x[2], y[2]]])
    De = np.linalg.det(mat)
    area = abs(De/2)
    areas[idt] = area
    
    b_grad = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])/De
    c_grad = -np.array([x[1]-x[2], x[2]-x[0], x[0]-x[1]])/De
    grad[idt] = np.array([b_grad, c_grad])

    Se = np.matrix(np.zeros((3,3)))
    for i in range(3):
        for j in range(3):
            Se[i, j] = (b_grad[i]*b_grad[j] + c_grad[i]*c_grad[j])*area*epsilon
            if not node_id[t[i]]: 
                if not node_id[t[j]]: 
                    S[index[t[i]], index[t[j]]] += Se[i, j]
                else:
                    B[index[t[i]]] -= Se[i, j]*X0[t[j]]


# also try cg or something for iterative solver
start = time.time() 
X = ssl.spsolve(S, B)
end = time.time() 
iter_start = end
Xiter, exit_code = ssl.bicg(S, B)
iter_end = time.time() 
# print(S)
# X = np.linalg.solve(S, B)
X0[unknown] = X

triang = tri.Triangulation(points[:, 0], points[:, 1], triangles)

plt.figure()
# plt.tripcolor(triang, X0, cmap='RdBu_r')
plt.tricontour(triang, X0, colors='black')
plt.tricontourf(triang, X0, levels=50, cmap='RdBu_r')
plt.colorbar(label="Potential")
plt.title("Electrostatic Potential")
plt.xlabel("x")
plt.xlabel("y")
plt.axis("equal")
plt.tight_layout()
plt.show()

E_per_element = np.array([-grad[idt] @ X0[tri] for idt, tri in enumerate(triangles)])

E_at_nodes = np.zeros(num_points, 2)
count = np.zeros(num_points) 

for tri, E in zip(triangles, E_per_element): 
    E_at_nodes[tri] += E
    count[tri] += 1 

for i in range(num_points): 
    E_at_nodes[i] /= count[i]

magnitude = np.zeros(num_points)
for i in range(num_points): 
    magnitude[i] = np.linalg.norm(E_at_nodes[i])
    E_at_nodes[i] /= magnitude[i]
plt.figure()
plt.quiver(points[:, 0], points[:, 1], E_at_nodes[:, 0], E_at_nodes[:, 1], 
           magnitude, scale=5e2,
           angles='xy', scale_units='xy', width=0.005, pivot='tail')
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.colorbar(label="Field Magnitude")
plt.tight_layout()
plt.show()

magnitude_on_elements = np.linalg.norm(E_per_element, axis=1)
We = 0.5*epsilon*np.sum(magnitude_on_elements**2 * areas)

V = INNERPOTENTIAL
C = 2*We/V**2
print(C)

C_th = 2*np.pi*epsilon/np.log(b/a)
print(C_th)