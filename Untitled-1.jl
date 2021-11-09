using DiffEqOperators
using SparseArrays
using LinearAlgebra
using Plots
using Sundials
using DifferentialEquations

##
x0 = 0.
x1 = 100
N = 101
dx = (x1-x0)/(N-1)
x = x0:dx:x1 
u = sin.(x)[2:end-1]

tspan=(0.,100.)
p=();

test = rand(N,N);
r0 = rand(N);

#gr()
#plot(x[2:end-1],u)


##
Δx = CenteredDifference{1}(2,2,dx,N,1);
Δy = CenteredDifference{2}(2,2,dx,N,1);
bc = GeneralBC([0.,1.],[0.,1.],dx,1)

# RobinBC((0.,1.,0.),(0.,1.,0.),dx,1)
# Neumann0BC(dx,1)
# PeriodicBC(eltype(u)) 
# Dirichlet0BC(eltype(u))

##


function basic!(dr,r,p,t)
    # mul!(Ayyu,Ayy,pi3k)
    # mul!(uAxx,pi3k,Axx)
    # mul!(Ayyv,Ayy,pten)
    # mul!(vAxx,pten,Axx)
    dr = Δx*bc*r #+ Δy*bc*r #PI3K
end

##


prob = ODEProblem(basic!,r0,tspan,p)
sol = solve(prob,Tsit5(),
save_everystep=false,
saveat = range(tspan[1],tspan[2], length=50)) ;
sol.retcode|>print
#SparseMatrixCSC(Δ)


##
plot!(x[2:end-1],w)
