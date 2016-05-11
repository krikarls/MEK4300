from fenics import *
import numpy as np

mesh = Mesh("medium_karman.xml")

# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, "Lagrange", 2)
Q = FunctionSpace(mesh, "Lagrange", 1)

# Define trial and test functions
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)

u0 = Function(V)
u1 = Function(V)
p0 = Function(Q)
p1 = Function(Q)
U  = 0.5*(u0 + u)

# Set parameter values
dt = 0.001/1.5; T = 8.0
rho = Constant(1.0); nu = Constant(1.0e-3); U_m = 1.5; mu = Constant(rho*nu); U_ = (2/3.)*U_m; D = .1

p_out = Constant(0)
u_in = Expression(('4.0*U_m*x[1]*(0.41-x[1])/(0.41*0.41)','0.0'),U_m=U_m)
zero_velocity = Expression(("0.0", "0.0"))

# Define boundary conditions
class NoSlip(SubDomain):
    def inside(self, x, on_boundry):
        return on_boundry

class Inflow(SubDomain):
    def inside(self, x, on_boundry):
        return (x[0] < 1e-6) and on_boundry

class Cylinder(SubDomain):
    def inside(self, x, on_boundry):
        return ( (x[0]-0.2)*(x[0]-0.2)+(x[1]-0.2)*(x[1]-0.2) < 0.051*0.051) and on_boundry

class Outflow(SubDomain):
    def inside(self, x, on_boundry):
        return (x[0]-2.2 < 1e-6) and (x[0]-2.2 > -1e-6) and on_boundry

mf = FacetFunction("size_t", mesh)
mf.set_all(4)

noslip = NoSlip(); noslip.mark(mf, 0)
inflow = Inflow(); inflow.mark(mf, 1)
cylinder = Cylinder(); cylinder.mark(mf, 3)
outflow = Outflow(); outflow.mark(mf, 2)
plot(mf, interactive=True)

BC_u  = DirichletBC(V, zero_velocity, mf, 0)
U_in  = DirichletBC(V, u_in, mf, 1)
BC_cyl  = DirichletBC(V, zero_velocity, mf, 3)
P_out = DirichletBC(Q, p_out, mf, 2)

u_bc = [BC_u,BC_cyl, U_in]
p_bc = [P_out]

def sigma(u,p):
    return 2.0*mu*sym(grad(u))-p*Identity(len(u))

n = FacetNormal(mesh); ds=ds[mf]
k = Constant(dt)
f = Constant((0, 0))

# Tentative velocity step
F1 = (rho/k)*inner(u-u0,v)*dx + rho*inner(grad(u0)*u0, v)*dx + inner(sigma((u0+u)/2.,p0), sym(grad(v)))*dx - dot(f, v)*dx + dot(p0*n, v)*ds - mu*inner(grad((u0+u)/2).T*n, v)*ds
a1 = lhs(F1)
L1 = rhs(F1)

# Correction of pressure
a2 = k*inner(grad(p), grad(q))*dx
L2 = k*inner(grad(p0), grad(q))*dx - rho*div(u1)*q*dx

# Correctioin of velocity
a3 = rho*inner(u, v)*dx
L3 = rho*inner(u1, v)*dx + k*inner(grad(p0-p1), v)*dx

# Create files for storing solution
ufile = File("results/velocity.pvd")
pfile = File("results/pressure.pvd")

# Matrix to store data
time_steps = int(round(T/dt))
data = np.zeros([time_steps+1,4])

# Time-stepping
t = 0; i = 0
while t < T:

    t += dt
    print "t =", t

    # Compute tentative velocity step
    solve(a1==L1, u1, bcs=u_bc)

    # Pressure correction
    solve(a2==L2,p1,bcs=p_bc)

    # Velocity correction
    solve(a3==L3,u1,bcs=u_bc)

    total_force = dot(-p1*Identity(2)+nu*(grad(u1)+grad(u1).T), n)
    Drag = -assemble(total_force[0]*ds(3))
    Lift = -assemble(total_force[1]*ds(3))

    Cd =  2*Drag/(U_**2*D) 
    Cl =  2*Lift/(U_**2*D) 

    dP = p1(np.array([0.15,0.20]))-p1(np.array([0.25,0.20]))

    data[i,0] = t 
    data[i,1] = Cd
    data[i,2] = Cl
    data[i,3] = dP

    # To avoid saving all time steps
    if i % 10 == 0: 
        ufile << u1
        pfile << p1

    i += 1

    # Move to next time step
    u0.assign(u1)
    p0.assign(p1)

np.savetxt('data.txt',data)






