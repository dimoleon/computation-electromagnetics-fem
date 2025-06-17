import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def local_stiffness_mass_grad_area(points, n1, n2, n3): 
    t = [n1, n2, n3]
    x = points[t][:, 0]
    y = points[t][:, 1]

    mat = np.array([[1, x[0], y[0]], 
                    [1, x[1], y[1]],
                    [1, x[2], y[2]]])
    De = np.linalg.det(mat)
    area = abs(De/2)

    b_grad = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])/De
    c_grad = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])/De
    grad = np.array([b_grad, c_grad])

    Se = np.zeros((3,3))
    for i in range(3): 
        for j in range(3): 
            Se[i, j] = (b_grad[i]*b_grad[j] + c_grad[i]*c_grad[j])*area
    
    Te = (area/12) * np.array([[2, 1, 1],
                               [1, 2, 1],
                               [1, 1, 2]])

    return Se, Te, grad, area

def plot_scalar(points, triangles, X0, title='Potential'): 
    triang = tri.Triangulation(points[:, 0], points[:, 1], triangles)

    plt.figure()
    plt.tricontour(triang, X0, colors='black')
    plt.tricontourf(triang, X0, levels=50, cmap='RdBu_r')
    plt.colorbar(label=title)
    plt.title(title)
    plt.xlabel("x")
    plt.xlabel("y")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

def calc_field_from_potential(potential, grad, triangles): 
    num_points = len(potential)

    E_per_element = np.array([-grad[idt] @ potential[tri] for idt, tri in enumerate(triangles)])

    E_at_nodes = np.zeros(num_points, 2)
    count = np.zeros(num_points) 

    for tri, E in zip(triangles, E_per_element): 
        E_at_nodes[tri] += E
        count[tri] += 1 

    for i in range(num_points): 
        E_at_nodes[i] /= count[i]

    return E_at_nodes, E_per_element

def plot_field(E_points, points, scale=5e2, title="Field Magnitude", color_title="V/m"):
    num_points = len(E_points)
    magnitude = np.zeros(num_points)
    for i in range(num_points): 
        magnitude[i] = np.linalg.norm(E_points[i])
        E_points[i] /= magnitude[i]
    plt.figure()
    plt.quiver(points[:, 0], points[:, 1], E_at_nodes[:, 0], E_at_nodes[:, 1], 
            magnitude, scale=scale,
            angles='xy', scale_units='xy', width=0.005, pivot='tail')
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.colorbar(label=color_title)
    plt.tight_layout()
    plt.show()

def calc_energy(E_elem, areas, epsilon): 
    magnitude_on_elements = np.linalg.norm(E_elem, axis=1)
    We = 0.5*epsilon*np.sum(magnitude_on_elements**2 * areas)

    return We

def calc_relative_error(val_theor, val_calc): 
    return 100*abs(val_theor - val_calc)/val_theor