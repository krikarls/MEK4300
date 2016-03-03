from dolfin import *

set_log_active(False)

def solver(mesh,deg):
	dpdx = Constant(-1)
	mu = Constant(100)
	A = 2

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
	beta = -dpdx/(2*sqrt(3)*A*mu)
	class u_exact(Expression):
		def eval(self,value,x):
			value[0] = beta*( (-x[1])-0.5*A*sqrt(3) )*(3*x[0]*x[0]-x[1]*x[1]) 

	u_e = project(u_exact(), V, bcs=noslip)
	u_error = errornorm(u_,u_e,degree_rise=0)

	# Compute the flux 
	mu = 1; dpdx = -1
	Q = assemble(u_*dx)
	Q_analytical = -dpdx*(A**4)*sqrt(3)/(320*mu)

	return mesh.hmin(), u_error


mesh = Mesh("course_triangle.xml")
deg = 1

E0=1; h0=1

print ' '
print 'Polynomial degree: ', deg
print '        h     ', '        E      ', '        r'
for n in range(0,4):
	h,E = solver(mesh,deg)
	if n > 0: 		# need two E/h pairs before convergence rate makes sense
		print h, E , ln(E/E0)/ln(h/h0)
	E0=E
	h0=h
	mesh = refine(mesh)

