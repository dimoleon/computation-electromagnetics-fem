import gmsh 
import meshio

def coaxial_mesh(a, b, refinements=0, model_name=f'coaxial_cross_section'): 
    gmsh.initialize() 
    gmsh.model.add(model_name)

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

    for _ in range(refinements): 
        gmsh.model.mesh.refine() 

    gmsh.write(f'{model_name}.msh')
    gmsh.finalize()

    msh = meshio.read(f'{model_name}.msh')

    return msh

def parallel_plates_mesh(w, t, d, domain_multiplier=5, refinements=0):
    model_name = f'parallel_plates_capacitor'
    gmsh.initialize() 
    gmsh.model.add(model_name)

    # domain defininitions
    domain_len = w*domain_multiplier

    p1 = gmsh.model.occ.addPoint(-domain_len/2, -domain_len/2, 0)
    p2 = gmsh.model.occ.addPoint(domain_len/2, -domain_len/2, 0)
    p3 = gmsh.model.occ.addPoint(domain_len/2, domain_len/2, 0)
    p4 = gmsh.model.occ.addPoint(-domain_len/2, domain_len/2, 0)

    l1 = gmsh.model.occ.addLine(p1, p2)
    l2 = gmsh.model.occ.addLine(p2, p3)
    l3 = gmsh.model.occ.addLine(p3, p4)
    l4 = gmsh.model.occ.addLine(p4, p1)

    domain_loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])

    # upper plate
    up1 = gmsh.model.occ.addPoint(-w/2, d/2, 0)
    up2 = gmsh.model.occ.addPoint(w/2, d/2, 0)
    up3 = gmsh.model.occ.addPoint(w/2, d/2 + t, 0)
    up4 = gmsh.model.occ.addPoint(-w/2, d/2 + t, 0)

    ul1 = gmsh.model.occ.addLine(up1, up2)
    ul2 = gmsh.model.occ.addLine(up2, up3)
    ul3 = gmsh.model.occ.addLine(up3, up4)
    ul4 = gmsh.model.occ.addLine(up4, up1)

    upper_plate_loop = gmsh.model.occ.addCurveLoop([ul1, ul2, ul3, ul4])

    # lower plate
    lp1 = gmsh.model.occ.addPoint(-w/2, -d/2 - t, 0)
    lp2 = gmsh.model.occ.addPoint(w/2, -d/2 - t, 0)
    lp3 = gmsh.model.occ.addPoint(w/2, -d/2, 0)
    lp4 = gmsh.model.occ.addPoint(-w/2, -d/2, 0)

    ll1 = gmsh.model.occ.addLine(lp1, lp2)
    ll2 = gmsh.model.occ.addLine(lp2, lp3)
    ll3 = gmsh.model.occ.addLine(lp3, lp4)
    ll4 = gmsh.model.occ.addLine(lp4, lp1)

    lower_plate_loop = gmsh.model.occ.addCurveLoop([ll1, ll2, ll3, ll4])

    # dielectric rectangle
    dl1 = ll3
    dl2 = gmsh.model.occ.addLine(lp4, up1)
    dl3 = ul1
    dl4 = gmsh.model.occ.addLine(up2, lp3)
    
    dielectric_loop = gmsh.model.occ.addCurveLoop([dl1, dl2, dl3, dl4])

    # surfaces 
    dielectric_surface = gmsh.model.occ.addPlaneSurface([dielectric_loop])
    air_surface = gmsh.model.occ.addPlaneSurface([domain_loop,
                                                  upper_plate_loop, lower_plate_loop,
                                                  dielectric_loop])
    
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(1, [l1, l2, l3, l4], 1, "domain_boundary")
    gmsh.model.addPhysicalGroup(1, [ul1, ul2, ul3, ul4], 2, "upper_plate")
    gmsh.model.addPhysicalGroup(1, [ll1, ll2, ll3, ll4], 3, "lower_plate")

    gmsh.model.addPhysicalGroup(2, [air_surface], 4, "air_surface")
    gmsh.model.addPhysicalGroup(2, [dielectric_surface], 5, "dielectric_surface")
    
    gmsh.model.mesh.generate(2)

    for _ in range(refinements):
        gmsh.model.mesh.refine() 

    
    gmsh.write(f'{model_name}.msh')
    gmsh.finalize()

    msh = meshio.read(f'{model_name}.msh')

    return msh

def circular_waveguide_mesh(a, refinements=0, model_name='circular_waveguide'): 
    gmsh.initialize() 
    gmsh.model.add(model_name)

    boundary = gmsh.model.occ.addCircle(0, 0, 0, a, 1)

    boundary_loop = gmsh.model.occ.addCurveLoop([boundary])
    domain = gmsh.model.occ.addPlaneSurface([boundary_loop])
    # domain = gmsh.model.occ.addDisk(0, 0, 0, a, a) 

    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(1, [boundary], 1, "boundary")
    gmsh.model.addPhysicalGroup(2, [domain], 1, "domain")

    gmsh.model.mesh.generate(2)

    for _ in range(refinements): 
        gmsh.model.mesh.refine()

    gmsh.write(f'{model_name}.msh')
    gmsh.finalize() 

    msh = meshio.read(f'{model_name}.msh')
    
    return msh

