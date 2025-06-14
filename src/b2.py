import numpy as np 
import meshes 
import scipy.sparse as ssp
import scipy.sparse.linalg as ssl
import matplotlib.pyplot as plt
import matplotlib.tri as tri

E0 = 1
f = 3e8
omega = 2*np.pi*f
wavelength = 1
a_sel = [wavelength/4, wavelength, 2.5*wavelength]
diffRa_sel = [wavelength/2, wavelength, 2*wavelength]
refinements = 3

a = a_sel[0]
diffRa = diffRa_sel[0]
R = a + diffRa

msh = meshes.coaxial_mesh(a=a, b=R, refinements=refinements, model_name='cylinder_scatter')

points = msh.points[:, :2]
edges = msh.cells_dict["line"]
triangles = msh.cells_dict["triangle"]

inner_nodes = set() 
outer_nodes = set() 

inner_lines_idxs = msh.cell_sets_dict["inner_boundary"]["line"]
outer_lines_idxs = msh.cell_sets_dict["outer_boundary"]["line"]

def local_stiffness_mass(p1, p2, p3): 
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

dirichlet_nodes = list(inner_nodes)

for t in triangles: 
    pass
    
 
