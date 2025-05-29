using Gridap
using GridapGmsh
using Gridap.Adaptivity
# using Gridap.CellData
using Gridap.Geometry
using Gmsh: gmsh, Gmsh
using Gridap.Visualization

include("constants.jl")
include("1a.jl")

model = GmshDiscreteModel("coaxial_cable.msh")
boun = BoundaryTriangulation(model)
# model = refine(model)
# model = refine(model)

trian = Triangulation(model)
writevtk(trian, "grok")

# p = get_cell_points(omega)

labels = get_face_labeling(model)
edges = get_faces(model, 1)
triangles = get_faces(model, 2)
# faces = get_face
# tags = get_tag_from_name(labels)
# tags = labels.d_to_dface_to_entity

# grid = get_grid(model)
# coords = get_node

coordinates = Gridap.Geometry.get_node_coordinates(model)