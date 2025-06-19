import numpy as np 
import meshes
import utils
import scipy.sparse as ssp
import scipy.sparse.linalg as ssl
import matplotlib.pyplot as plt 
import matplotlib.tri as tri
from scipy.constants import epsilon_0

width = 4e-2
thickness = 0.2e-2 
dist = 1e-2
V = 100
er = 2.2 
epsilon = er*epsilon_0
domain_factor = 5
refinements = 2

UPPERPOTENTIAL = V/2 
LOWERPOTENTIAL = -V/2 

def inside_dielectric(x, y): 
    return abs(y) < dist/2 + thickness/2 and abs(x) < width/2 + 1e-7
        

def a2(refs=refinements, domain_factor=domain_factor, plots=True):
    msh = meshes.parallel_plates_mesh(width, thickness, dist, domain_factor, refs)

    points = msh.points[:, :2]
    edges = msh.cells_dict["line"]
    triangles = msh.cells_dict["triangle"]

    upper_lines_idxs = msh.cell_sets_dict["upper_plate"]["line"]
    lower_lines_idxs = msh.cell_sets_dict["lower_plate"]["line"]

    # air_triangles_idxs = msh.cell_sets_dict["air_surface"]["triangle"]
    # dielectric_triangles_idxs = msh.cell_sets_dict["dielectric_surface"]["triangle"]
    dielectric_triangles_idxs = []
    for idt, t in enumerate(triangles): 
        count = 0
        for n in t: 
            if inside_dielectric(*points[n]):
                count += 1
        if count == 3: 
            dielectric_triangles_idxs.append(idt)

    dielectric_triangles_idxs = np.array(dielectric_triangles_idxs) 

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
    stiffnesses = np.empty(num_trian, dtype=object)

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

    S = ssp.csc_matrix(S)
    X = ssl.spsolve(S, B)
    X0[unknown] = X

    E_per_element, E_at_nodes = utils.calc_field_from_potential(X0, grads, triangles)

    E_elem_dielectric = E_per_element[dielectric_triangles_idxs]
    areas_dielectric = areas[dielectric_triangles_idxs]

    # We_calc = utils.calc_energy(E_elem_dielectric, areas_dielectric, epsilon)
    We_calc = 0
    for idt, t in enumerate(triangles[dielectric_triangles_idxs]): 
        We_calc += X0[t].T @ stiffnesses[dielectric_triangles_idxs][idt] @ X0[t]
    We_calc *= 0.5*epsilon
    capacitance_calc = 2*We_calc/V**2

    capacitance_theoretical = epsilon*width/dist

    rel_err = utils.calc_relative_error(capacitance_theoretical, capacitance_calc)

    print(f"Theoretical capacitance: \t{capacitance_theoretical} F/m")
    print(f"Calculated capacitance: \t{capacitance_calc} F/m")
    print(f"Relative Error: \t\t{rel_err:4f} %")

    if plots: 
        utils.plot_scalar(points, triangles, X0)
        utils.plot_field(E_at_nodes, points, scale=2e2, step=3)

for ref in range(1, 4): 
    ref_plurar = 'refinement' if ref == 1 else 'refinements'
    print(f"\nProblem A2(a) with {ref} " + ref_plurar)
    a2(refs=ref, plots=True)

for domf in [3, 5, 7, 9]: 
    print(f"\nProblem A2(b) with {domf} domain multiplier")
    a2(refs=refinements, domain_factor=domf, plots=False)
