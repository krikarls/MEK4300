from dolfin import *
import mshr as mshh

## Generate mesh with different density ##
elips = mshh.Ellipse(Point(0,0,0), 1.0, 0.5, 10)
mesh2 = mshh.generate_mesh(elips, 5)

elips = mshh.Ellipse(Point(0,0,0), 1.0, 0.5, 20)
mesh3 = mshh.generate_mesh(elips, 10)

elips = mshh.Ellipse(Point(0,0,0), 1.0, 0.5, 40)
mesh4 = mshh.generate_mesh(elips, 20)

elips = mshh.Ellipse(Point(0,0,0), 1.0, 0.5, 80)
mesh5 = mshh.generate_mesh(elips, 40)

set_log_active(False)

def solver(mesh,deg):
	dpdx = Constant(-1)
	mu = Constant(100)
	A = 1.0   # ellipse width
	B = 0.5  # ellipse hight

	V = FunctionSpace(mesh, "Lagrange", deg)
	u = TrialFunction(V)
	v = TestFunction(V)

	# Mark boundary subdomians
	class Sides(SubDomain):
		def inside(self, x, on_boundry):
			return on_boundry

	side = Sides()

	mf = FacetFunction("size_t", mesh)
	mf.set_all(2)

	side.mark(mf, 1)
	noslip = DirichletBC(V, Constant(0), mf, 1)

	a = inner( grad(u), grad(v) )*dx 
	L = -1.0/mu*dpdx*v*dx

	u_ = Function(V)
	solve(a == L, u_, bcs=noslip)

	# Implement analytical solution
	beta = -dpdx*(1.0/(2*mu))*(A**2*B**2)/(A**2+B**2)
	class u_exact(Expression):
		def eval(self,value,x):
			value[0] = beta*(1-x[0]*x[0]/(A*A)-x[1]*x[1]/(B*B))

	#u_e = project(u_exact(), V, bcs=noslip)
	u_e = interpolate(u_exact(), V)
	u_error = errornorm(u_,u_e,degree_rise=0)

	# Compute the flux 
	mu = 1; dpdx = -1
	Q = assemble(u_*dx)
	Q_analytical =  -pi/(4*mu)*dpdx*(A**3*B**3)/(A**2+B**2)

	return mesh.hmin(), u_error

deg = 1

h = [0,0,0,0,0]; E = [0,0,0,0,0]

#h[0],E[0] = solver(mesh1,deg)
h[0],E[0] = solver(mesh2,deg)
h[1],E[1] = solver(mesh3,deg)
h[2],E[2] = solver(mesh4,deg)
h[3],E[3] = solver(mesh5,deg)

print 'Polynomial degree = ', deg
print '        h     ', '        E      ', '        r'
for i in range(1,4):
	print h[i], E[i] , ln(E[i]/E[i-1])/ln(h[i]/h[i-1])
