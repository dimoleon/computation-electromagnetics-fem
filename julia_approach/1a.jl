using Gridap
using GridapGmsh
using Gmsh: gmsh, Gmsh

include("constants.jl")

# Initialize Gmsh
Gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.model.add("coaxial")
# gmsh.model.setFactory("OpenCASCADE")


# Add outer and inner disks
outer_circle = gmsh.model.occ.add_circle(0.0, 0.0, 0.0, bmm, 1)
inner_circle = gmsh.model.occ.add_circle(0.0, 0.0, 0.0, amm, 2)

outer_loop = gmsh.model.occ.add_curve_loop([1], 3)
inner_loop = gmsh.model.occ.add_curve_loop([2], 4)

# Subtract inner disk from outer disk to get annulus
surface = gmsh.model.occ.add_plane_surface([3, 4], 5)
# surface = gmsh.model.occ.addPlaneSurface([outer_loop, inner_loop])  # Outer loop with inner loop as hole

# Synchronize gmsh model
gmsh.model.occ.synchronize()

gmsh.model.addPhysicalGroup(1, [1], 6, "outer_conductor")
gmsh.model.addPhysicalGroup(1, [2], 7, "inner_conductor")
gmsh.model.addPhysicalGroup(2, [5], 8, "dielectric")

# Define physical groups (optional but useful for BCs)
# gmsh.model.addPhysicalGroup(2, [ent[2] for ent in annulus], 1)
# gmsh.model.setPhysicalName(2, 1, "air")

# gmsh.model.mesh.setTransfiniteCurve(inner_circle, 50)  # 50 elements on inner circle
# gmsh.model.mesh.setTransfiniteCurve(outer_circle, 80)  # 80 elements on outer circle

# Generate mesh
gmsh.model.mesh.generate(2)

# Save mesh to file (optional)
gmsh.write("coaxial_cable.msh")

Gmsh.finalize()