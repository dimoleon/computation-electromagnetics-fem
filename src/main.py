import time
import gmsh
import numpy as np
import meshio
import scipy.sparse as ssp
import scipy.sparse.linalg as ssl
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.constants import epsilon_0


Z = 50
er = 1
epsilon = epsilon_0 * er
b = 1.75
a = b*np.exp(-Z*np.sqrt(er)/60)
INNERPOTENTIAL = 1.0
OUTERPOTENTIAL = 0.0

gmsh.initialize()
gmsh.model.add("coaxial_cross_section")

inner = gmsh.model.occ.addCircle(0, 0, 0, a, 1) 
outer = gmsh.model.occ.addCircle(0, 0, 0, b, 2)

inner_loop = gmsh.model.occ.addCurveLoop([1], 3)
outer_loop = gmsh.model.occ.addCurveLoop([2], 4)
surface = gmsh.model.occ.addPlaneSurface([4, 3], 5)

gmsh.model.occ.synchronize()

inner_tag = gmsh.model.addPhysicalGroup(1, [1], 6, "inner_boundary")
outer_tag = gmsh.model.addPhysicalGroup(1, [2], 7, "outer_boundary")
dielectric_tag = gmsh.model.addPhysicalGroup(2, [5], 8, "dielectric")

gmsh.model.mesh.generate(2)
gmsh.write("coaxial.msh")
gmsh.finalize()

msh = meshio.read("coaxial.msh")
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
# S = np.matrix(np.zeros((Nf, Nf)))
B = np.zeros(Nf)

for t in triangles: 
    x = [p[0] for p in points[t]]
    y = [p[1] for p in points[t]]
    mat = np.matrix([[1, x[0], y[0]], 
                    [1, x[1], y[1]],
                    [1, x[2], y[2]]])
    De = np.linalg.det(mat)
    area = abs(De/2)
    
    b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])/De
    c = -np.array([x[1]-x[2], x[2]-x[0], x[0]-x[1]])/De

    Se = np.matrix(np.zeros((3,3)))
    for i in range(3):
        for j in range(3):
            Se[i, j] = (b[i]*b[j] + c[i]*c[j])*area*epsilon
            if not node_id[t[i]]: 
                if not node_id[t[j]]: 
                    S[index[t[i]], index[t[j]]] += Se[i, j]
                else:
                    B[index[t[i]]] -= Se[i, j]*X0[t[j]]


# also try cg or something for iterative solver
X = np.array(ssl.spsolve(S, B))
# print(S)
# X = np.linalg.solve(S, B)
X0[unknown] = X

triang = tri.Triangulation(points[:, 0], points[:, 1], triangles)

plt.figure()
plt.tricontourf(triang, X0, levels=50, cmap="viridis")
plt.colorbar(label="Potential")
plt.title("Electrostatic Potential")
plt.xlabel("x")
plt.xlabel("y")
plt.axis("equal")
plt.tight_layout()
plt.show()
