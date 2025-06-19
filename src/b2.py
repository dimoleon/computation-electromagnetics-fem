import numpy as np 
import meshes 
import utils
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


a_ex = a_sel[1]
diffRa_ex = diffRa_sel[2]
R_ex = a_ex + diffRa_ex

def b2(a=a_ex, R=R_ex):
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
        Se, Te, _, _  = utils.local_stiffness_mass_grad_area(points, *t)
        for i in range(3): 
            for j in range(3): 
                    S[t[i], t[j]] += Se[i, j]
                    T[t[i], t[j]] += Te[i, j]

    A = S - k0**2 * T

    for e in edges[outer_lines_idxs]: 
        id1, id2 = e[0], e[1]
        length = np.linalg.norm(points[id1]-points[id2])

        coefficient = (-1j * k0 - 1/(2*R)) 
        mass_mat = (length/6) * np.array([[2, 1], 
                                        [1, 2]])
        ABCe = coefficient * mass_mat
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

    utils.plot_scalar(points, triangles, E_total,
                    title=f"Total Electric Field [a, R] = [{a}, {R}]m", color_title="Ez (V/m)",
                    equilines=False)

for a in a_sel: 
    for diff in diffRa_sel: 
        R = a + diff
        b2(a=a, R=R)