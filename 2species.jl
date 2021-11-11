#line
using DiffEqOperators
using SparseArrays
using LinearAlgebra
using Plots
using Sundials
using DifferentialEquations

##
x0 = 0.
x1 = 5.
N = 101
dx = (x1-x0)/(N-1)
x = x0:dx:x1 
u = sin.(x)[2:end-1]

tspan=(0.,10)
p=(0.001,0.1);


#gr()
#plot(x[2:end-1],u)


## Operators
Δx = CenteredDifference{1}(2,2,dx,N,1);
Δy = CenteredDifference{2}(2,2,dx,N,1);
bcx = Neumann0BC(dx,1); # left, right
bcy = Neumann0BC(dx,1); # down, up

# GeneralBC([1.,1.],[.5,1.],dx,1) # α1 + α2*u + α3*u'... = 0
# RobinBC((0.,1.,0.),(0.,1.,0.),dx,1)
# Neumann0BC(dx,1)
# PeriodicBC(eltype(u)) 
# Dirichlet0BC(eltype(u))


## Problem 

dxA = zeros(N,N)
dyA = zeros(N,N)
ΔA  = zeros(N,N)

dxB = zeros(N,N)
dyB = zeros(N,N)
ΔΒ  = zeros(N,N)

function basic!(dr,r,p,t)
    d1,d2 = p
    A = @view r[:,:,1]
    B = @view r[:,:,2]
    dA = @view dr[:,:,1]
    dB = @view dr[:,:,2]
    mul!(dxA,Δx,bcx*A)
    mul!(dyA,Δy,(bcy*A')')
    mul!(dxB,Δx,bcx*B)
    mul!(dyB,Δy,(bcy*B')')
    @. ΔA = dxA + dyA #Δx*bcx*A + Δy*(bcy*A')'
    @. ΔΒ = dxB + dyB#Δx*bcx*A + Δy*(bcy*A')'
    @. dA = d1*ΔA + A - A*B
    @. dB = d2*ΔΒ + A*B - B
end

## Initial Condition

r0 = zeros(N,N,2);
r0[:,:,1] .= rand.();
r0[:,:,2] .= rand.();
##


prob = ODEProblem(basic!,r0,tspan,p)
sol = solve(
    prob,
    CVODE_BDF(linear_solver=:GMRES),
    save_everystep=false,
    saveat = range(tspan[1],tspan[2], length=100)
    ) ;
sol.retcode|>println
#SparseMatrixCSC(Δ)


##

clims = (0,1);
anim = @animate for i=1:size(sol.t)[1]
    gr()
    data1 = @view sol.u[i][:,:,1]
    data2 = @view sol.u[i][:,:,2]
    p1=heatmap(1:size(data1,1),
        1:size(data1,2), data1,
        c=cgrad([:blue, :white,:red, :yellow]),
        xlabel="x values", ylabel="y values",
        clims = clims,
        title="A")
    p2=heatmap(1:size(data2,1),
        1:size(data2,2), data2,
        c=cgrad([:blue, :white,:red, :yellow]),
        xlabel="x values", ylabel="y values",
        clims = clims,
        title="B")

    fig=plot(p1,p2,layout=grid(1,2)) 
    #savefig(fig,"results/plot"*"s$N"*".png")
end
gif(anim, "test.gif", fps = 20);