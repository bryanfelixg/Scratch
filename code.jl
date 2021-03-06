#line
using DiffEqOperators
using SparseArrays
using LinearAlgebra
using Plots
using Sundials
using DifferentialEquations

## Spatial domain and number of nodes
x0 = 0.
x1 = 1.
N = 100
dx = (x1-x0)/(N-1)
x = x0:dx:x1 

## Time Span
tspan=(0.,200.)

## Parameters
p  = (1.0 , .2, 1.0 , 0.8 , 2. , 1., 0.0001, .05) ;


## Operators
Δx = CenteredDifference{1}(2,2,dx,N,1) ;
Δy = CenteredDifference{2}(2,2,dx,N,1) ;
bcx = Neumann0BC(dx,1); # left, right
bcy = Neumann0BC(dx,1); # down, up

Axx = Array(Δx*bcx)[1];
Ayy = Array(Δy*bcy)[1] ;


## Problem 

# Allocation
dx_pi3k = zeros(N,N)
dy_pi3k = zeros(N,N)
Δpi3k   = zeros(N,N)

dx_pten = zeros(N,N)
dy_pten = zeros(N,N)
Δpten  = zeros(N,N)

# ODE function
function basic!(dr,r,p,t)
    k1,k2,k3,k4,k5,k6,D1,D2 = p
    pi3k = @view r[:,:,1]
    pten = @view r[:,:,2]
    pip3 = @view r[:,:,3]
    d_pi3k = @view dr[:,:,1]
    d_pten = @view dr[:,:,2]
    d_pip3 = @view dr[:,:,3]
    mul!(dx_pi3k,Axx,pi3k)
    mul!(dy_pi3k,pi3k,Ayy)
    mul!(dx_pten,Axx,pten)
    mul!(dy_pten,pten,Ayy)

    @. Δpi3k = dx_pi3k+dy_pi3k
    @. Δpten = dx_pten+dy_pten 
    @. d_pi3k .= D1*Δpi3k + k4*(1-pi3k)*pip3 - k3*pi3k*(1-pip3)
    @. d_pten .= D2*Δpten + k1*(1-pip3)*(1-pten) - k2*pip3*pten
    @. d_pip3 .= k5*(1-pip3)*pi3k - k6*pip3*pten
end

## Initial Condition

r0 = zeros(N,N,3);
r0[:,:,1] .= 0. # PI3K
r0[:,:,2] .= .999; # PTEN
r0[:,:,3] .= 0. #PIP3

r0[45:55,45:55,1] .= 0.99 # "Pulse of PI3K"

## Solver

prob = ODEProblem(basic!,r0,tspan,p);

sol =solve(
    prob,
    CVODE_BDF(linear_solver=:GMRES),
    save_everystep=false,
    saveat = range(tspan[1],tspan[2], length=100)
    ) ;


## Plot

clims = (0,1);
colors = cgrad([:blue,:yellow,:red]);
anim = @animate for i=1:size(sol.t)[1]
    gr()
    data1 = @view sol.u[i][:,:,1]
    data2 = @view sol.u[i][:,:,2]
    data3 = @view sol.u[i][:,:,3]
    p1=heatmap(x,x, 
        data1,
        c=colors,
        xlabel="x values", ylabel="y values",
        clims = clims,
        title="PI3K t=$(round(sol.t[i];digits=1))")
    p2=heatmap(x,x,
        data2,
        c=colors,
        xlabel="x values", ylabel="y values",
        clims = clims,
        title="PTEN")
    p3=heatmap(x,x, 
        data3,
        c=colors,
        xlabel="x values", ylabel="y values",
        clims = clims,
        title="PIP3")
    p4=heatmap(x,x, 
        1.0.-data3,
        c=colors,
        xlabel="x values", ylabel="y values",
        clims = clims,
        title="PIP2")

    fig=plot(p1,p2,p3,p4,
        layout=grid(2,2),
        size=(800,600)) 
end
##
gif(anim, "movie.gif", fps = 15); 