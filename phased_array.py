# # Helmholtz equation for acoustic phased arrays using FEniCSx
#
# Copyright (C) 2023 Prakash Manandhar
#
# Phase arrays inspired by and,
# adapted from/combining approaches in: https://gitlab.com/cfd-pizca/hef-acoustics,
#  and https://github.com/FEniCS/dolfinx/blob/main/python/demo/demo_helmholtz.py
#  and https://fenicsproject.discourse.group/t/pointsource-in-dolfinx/8337/13
#  and https://fenicsproject.discourse.group/t/is-there-a-way-to-implement-perfectly-matched-layers/4951 
#  and https://github.com/samuelpgroth/waves-fenicsx/blob/master/frequency/demo0_scattering_by_circle.py 
# +

import numpy as np

import ufl
from dolfinx.fem import Function, FunctionSpace, assemble_scalar, form, locate_dofs_geometrical
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_rectangle
from ufl import dx, grad, inner

from mpi4py import MPI
from petsc4py import PETSc

#frequency, speed, and wavelength

f0 = 800.0 # Hz
c0 = 330.0 # m/s
lambda0 = c0/f0 # m

# wavenumber
k0 = 2 * np.pi / lambda0

print (f"f0 = {f0:0.1f} Hz, c0 = {c0:0.1f} m/s, lambda0 = {lambda0:0.4g} m, k0 = {k0:0.4g} m")

# phased array settings
theta_steering_deg = np.arange(-75, 75, 1.0) # degrees
phased_pitch = lambda0/16.0
num_phased_elements = 64 # integer
aperature = phased_pitch * (num_phased_elements - 1)
near_field = aperature*aperature / ( 4*lambda0 )
print(f"pitch = {phased_pitch} m")
print(f"Aperatture = {aperature} m")
print(f"Nearfield = {near_field} m")

dim_x = 20*lambda0      # width of computational domain

'''                   Adiabatic absorber settings                           '''
# The adiabatic absorber is a PML-type layer in which absorption is used to
# attenutate outgoing waves. Adiabatic absorbers aren't as perfect as PMLs so
# must be slightly wider: typically 2-5 wavelengths gives adequately small
# reflections.
d_absorb = 2 * lambda0    # depth of absorber

'''                   Adiabatic absorber settings                           '''
# The adiabatic absorber is a PML-type layer in which absorption is used to
# attenutate outgoing waves. Adiabatic absorbers aren't as perfect as PMLs so
# must be slightly wider: typically 2-5 wavelengths gives adequately small
# reflections.
d_absorb = 2 * lambda0    # depth of absorber

# Increase the absorption within the layer gradually, as a monomial:
# sigma(x) = sigma_0 * x^d; choices d=2,3 are popular choices.
deg_absorb = 2    # degree of absorption monomial

# The constant sigma_0 is chosen to achieve a specified "round-trip" reflection
# of a wave that through the layer, reflects and returns back into the domain.
# See Oskooi et al. (2008) for more details.
RT = 1.0e-6       # round-trip reflection
sigma0 = -(deg_absorb + 1) * np.log(RT) / (2.0 * d_absorb)


def adiabatic_layer(x):
    '''          Contribution to wavenumber k in absorbing layers          '''
    # In absorbing layer, have k = k0 + 1j * sigma
    # => k^2 = (k0 + 1j*sigma)^2 = k0^2 + 2j*sigma*k0 - sigma^2
    # Therefore, the 2j*sigma - sigma^2 piece must be included in the layer.

    # Find borders of width d_absorb in x- and y-directions
    in_absorber_x = (np.abs(x[0]) >= dim_x/2 - d_absorb)
    in_absorber_y = (      -x[1]  >= dim_x/2 - d_absorb) # bottom wall is absorbing reflecting, top reflecting

    # Function sigma_0 * x^d, where x is depth into adiabatic layer
    sigma_x = sigma0 * ((np.abs(x[0])-(dim_x/2-d_absorb))/d_absorb)**deg_absorb
    sigma_y = sigma0 * ((np.abs(x[1])-(dim_x/2-d_absorb))/d_absorb)**deg_absorb

    # 2j*sigma - sigma^2 in absorbing layers
    x_layers = in_absorber_x * (2j * sigma_x * k0 - sigma_x**2)
    y_layers = in_absorber_y * (2j * sigma_y * k0 - sigma_y**2)

    return x_layers + y_layers


# approximation space polynomial degree
deg = 1

# number of elements in each direction of msh
n_elem = 128


msh = create_rectangle(MPI.COMM_WORLD, [np.array([-dim_x/2, -dim_x/2]), np.array([dim_x/2, dim_x/2])], 
                               [n_elem, n_elem])
n = ufl.FacetNormal(msh)

# This implementation relies on the complex mode of dolfin-x
if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
    print('This demo only works with PETSc-complex')
    exit()


def simulate_steering_anlge(theta_deg, xdmffile):
    print(f"Solving for steering angle {theta_deg} deg ...")
    theta_steering = theta_deg*np.pi/180.0 # radians
    d_phi = 2*np.pi*phased_pitch*np.sin( theta_steering )/lambda0

    # Test and trial function space
    V = FunctionSpace(msh, ("Lagrange", deg))

    # Interpolate absorbing layer piece of wavenumber k_absorb onto V
    k_absorb = Function(V)
    k_absorb.interpolate(adiabatic_layer)

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    print("Applying Point Sources ...")
    f = Function(V)
    total_phase_angle = 0
    for i_loc in range(-int(num_phased_elements/2), int(num_phased_elements/2)):
        phased_dx = i_loc * phased_pitch
        A = PETSc.ScalarType(np.sin(total_phase_angle) + np.cos(total_phase_angle)*1j)
        dofs = locate_dofs_geometrical(V,  lambda x: np.isclose(x.T, [phased_dx, 3.0, 0.0], atol=0.1).all(axis=1))
        #print(f"dx = {phased_dx}")
        #print(dofs)25
        f.x.array[dofs] = A
        total_phase_angle += d_phi

    a = inner(grad(u), grad(v)) * dx - k0**2 * inner(u, v) * dx - k_absorb * inner(u, v) * dx
    L = inner(f, v) * dx

    # Compute solution
    uh = Function(V)
    uh.name = "u"
    problem = LinearProblem(a, L, u=uh, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    problem.solve()

    file.write_function(uh, theta_deg)
# -

if __name__ == "__main__":
    with XDMFFile(MPI.COMM_WORLD, f"out_phased_array/phased_array_near_wall_reflect_{num_phased_elements}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as file:
        file.write_mesh(msh)
        for th in theta_steering_deg:
            simulate_steering_anlge(th, file)
