geo_content = """
// Define geometric parameters
center_x = 0;
center_y = 0;
center_z = 0;
radius_inner = $(amm); // mm
radius_outer = $(bmm); // mm

// Define points
// Center point
Point(1) = {center_x, center_y, center_z, 1.0};

// Points for inner circle
Point(2) = {center_x + radius_inner, center_y, center_z, 1.0};
Point(3) = {center_x - radius_inner, center_y, center_z, 1.0};

// Points for outer circle
Point(4) = {center_x + radius_outer, center_y, center_z, 1.0};
Point(5) = {center_x - radius_outer, center_y, center_z, 1.0};

// Define curves (circles)
// Inner circle
Circle(1) = {2, 1, 3}; // Start point, center point, end point
Circle(2) = {3, 1, 2}; // Completing the circle

// Outer circle
Circle(3) = {4, 1, 5}; // Start point, center point, end point
Circle(4) = {5, 1, 4}; // Completing the circle

// Define curve loops
Curve Loop(1) = {1, 2}; // Inner circle loop
Curve Loop(2) = {3, 4}; // Outer circle loop

// Define surfaces (disks)
Plane Surface(1) = {1}; // Inner disk
Plane Surface(2) = {2}; // Outer disk

// Boolean operation: Subtract inner disk from outer disk
// This creates the dielectric region of the coaxial cable
// The result will be a new surface with tag 3 (by default, if not specified)
BooleanDifference(3) = {Surface{2}; Delete;}{Surface{1}; Delete;};

// Define physical groups for boundaries and the surface
// Physical groups are used to assign properties in solvers

// Boundary of the inner conductor (the curve loop of the inner circle)
Physical Curve("inner_conductor_boundary", 1) = {1, 2};

// Boundary of the outer conductor (the curve loop of the outer circle)
Physical Curve("outer_conductor_boundary", 2) = {3, 4};

// Surface of the dielectric material
Physical Surface("dielectric", 3) = {3};

// Optional: Define mesh size (uncomment and adjust as needed)
// Mesh.CharacteristicLengthMin = 0.1;
// Mesh.CharacteristicLengthMax = 0.2;
"""

geo_filename = "coaxial.geo"
open(geo_filename, "w") do f
    write(f, geo_content)
end

msh_filename = "coaxial.msh"
run(`gmsh -2 $geo_filename -format msh2 -o $msh_filename`)
run(`gmsh $msh_filename -0`)
run(`gmsh coaxial.msh -format vtk -save`)