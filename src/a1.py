import time
import numpy as np
import meshes 
import utils
import scipy.sparse as ssp
import scipy.sparse.linalg as ssl
from scipy.constants import epsilon_0


Z = 50
er = 1
b = 0.0175
a = b*np.exp(-Z*np.sqrt(er)/60)
epsilon = er*epsilon_0
INNERPOTENTIAL = 1.0
OUTERPOTENTIAL = 0.0
refinements = 4

def a1(refinements=refinements): 
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

    Nf = len(unknown)

    index = np.full(num_points, -1)
    index[unknown] = np.arange(Nf)

    node_id = np.full(num_points, False)
    node_id[known] = True

    S = ssp.lil_matrix((Nf, Nf))
    B = np.zeros(Nf)

    grads = np.empty(num_trian, dtype=object)
    areas = np.zeros(num_trian)
    stiffnesses = np.zeros(num_trian, dtype=object)

    for idt, t in enumerate(triangles): 
        Se, _, grad, area = utils.local_stiffness_mass_grad_area(points, *t)

        areas[idt] = area
        grads[idt] = grad
        stiffnesses[idt] = Se

        for i in range(3):
            for j in range(3):
                if not node_id[t[i]]: 
                    if not node_id[t[j]]: 
                        S[index[t[i]], index[t[j]]] += Se[i, j]
                    else:
                        B[index[t[i]]] -= Se[i, j]*X0[t[j]]


    start_direct = time.time() 
    X = ssl.spsolve(S, B)
    end_direct = time.time() 
    direct_time = end_direct - start_direct

    start_iter = time.time()
    Xiter, exit_code = ssl.bicg(S, B)
    end_iter = time.time() 
    iter_time = end_iter - start_iter

    X0[unknown] = X

    E_per_element, E_at_nodes = utils.calc_field_from_potential(X0, grads, triangles)

    # We_calc = utils.calc_energy(E_per_element, areas, epsilon)
    We_calc = 0
    for idt, t in enumerate(triangles): 
        We_calc += X0[t].T @ stiffnesses[idt] @ X0[t]
    We_calc *= 0.5*epsilon

    V = INNERPOTENTIAL - OUTERPOTENTIAL
    capacitance_calc = 2*We_calc/V**2

    capacitance_theoretical = 2*np.pi*epsilon/np.log(b/a)

    rel_err = utils.calc_relative_error(capacitance_theoretical, capacitance_calc)

    print(f"Degrees of Freedom: \t\t{Nf}")
    print(f"Time direct solver: \t\t{direct_time} s")
    print(f"Time iterative solver: \t\t{iter_time} s")
    print(f"Theoretical capacitance: \t{capacitance_theoretical} F/m")
    print(f"Calculated capacitance: \t{capacitance_calc} F/m")
    print(f"Relative Error: \t\t{rel_err:4f} %")

    utils.plot_scalar(points, triangles, X0)
    utils.plot_field(E_at_nodes, points, scale=5e2)

for r in range(5): 
    ref_plurar = 'refinement' if r == 1 else 'refinements'
    print(f"Problem A1 with {r} " + ref_plurar)
    a1(refinements=r)