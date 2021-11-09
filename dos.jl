#=
Looking at 
du/dt = D1(A_yy u + u A_xx) 
dv/dt = D2(A_yy v + v A_xx) 
Dirichlet Conditions
=#
using LinearAlgebra
using DifferentialEquations
#using BenchmarkTools
using Sundials
using Plots
using DiffEqOperators
using Distributed

##
#include("sampling.jl")

## Some constants
tspan = (0.0,1600.0)

N  = 100
x0 = 0.
x1 = 100.
dx = (x1-x0)/(N)
Ω = range(x0,x1,length=N+1)

p  = (5.0 , 5.0, 1.0 , 10.0 , 1.0 , 0.04, .8 , .2) # D1,D2

#p  = (1.0 , 2.0 , 5.0 , 0.5 , 0.58 , 1.0, .1 , 1.0)  # D1,D2

Axx = Array(Tridiagonal(
    [1.0 for i in 1:N-1],
    [-2.0 for i in 1:N],
    [1.0 for i in 1:N-1]
    ))
    
Ayy = copy(Axx)
Axx[2,1] = 2.0 
Axx[end-1,end] = 2.0
Ayy[1,2] = 2.0
Ayy[end,end-1] = 2.0
##

#Some allocation
Dpi3k = zeros(N,N)
Dpten = copy(Dpi3k)

Ayyu = zeros(N,N)
Ayyv = copy(Ayyu)

uAxx = zeros(N,N)
vAxx = copy(uAxx)

function basic!(dr,r,p,t)
    k1,k2,k3,k4,k5,k6,D1,D2 = p
    pi3k = @view r[:,:,1]
    pten = @view r[:,:,2]
    pip3 = @view r[:,:,3]
    dpi3k = @view dr[:,:,1] 
    dpten = @view dr[:,:,2]
    dpip3 = @view dr[:,:,3]
    mul!(Ayyu,Ayy,pi3k)
    mul!(uAxx,pi3k,Axx)
    mul!(Ayyv,Ayy,pten)
    mul!(vAxx,pten,Axx)
    @. Dpi3k = D1*(Ayyu + uAxx) #PI3K
    @. Dpten = D2*(Ayyv + vAxx)
    @. dpi3k = D1*Dpi3k + k3*pip3*(1-pi3k) - k4*pten*pi3k
    @. dpten = D2*Dpten + k1*(1-pip3)*(1-pten) - k2*pi3k*pten 
    @. dpip3 = k5*pip3*pi3k*(1-pip3) - k6*pip3*pten
end

## Initial Conditions
#D1,D2 = p
r0 = zeros(N,N,3);
r0[:,:,1] .= rand.();
r0[:,:,2] .= rand.();
r0[:,:,3] .= rand.(); #vss 

## Problem
prob = ODEProblem(basic!,r0,tspan,p)
sol = solve(prob,CVODE_BDF(linear_solver=:GMRES),
progress=true,
save_everystep=false,
saveat = range(tspan[1],tspan[2], length=50)) ;
sol.retcode|>print

##
clims = (0,1);
anim = @animate for i=1:size(sol.t)[1]
    gr()
    data1 = @view sol.u[i][:,:,1]
    data2 = @view sol.u[i][:,:,2]
    data3 = @view sol.u[i][:,:,3]
    p1=heatmap(1:size(data1,1),
        1:size(data1,2), data1,
        c=cgrad([:blue, :white,:red, :yellow]),
        xlabel="x values", ylabel="y values",
        clims = clims,
        title="PI3K")
    p2=heatmap(1:size(data2,1),
        1:size(data2,2), data2,
        c=cgrad([:blue, :white,:red, :yellow]),
        xlabel="x values", ylabel="y values",
        clims = clims,
        title="PTEN")
    p3=heatmap(1:size(data3,1),
        1:size(data3,2), data3,
        c=cgrad([:blue, :white,:red, :yellow]),
        xlabel="x values", ylabel="y values",
        clims = clims,
        title="PIP3")
    p4=heatmap(1:size(data3,1),
        1:size(data3,2), 1.0.-data3,
        c=cgrad([:blue, :white,:red, :yellow]),
        xlabel="x values", ylabel="y values",
        clims = clims,
        title="PIP2")

    fig=plot(p1,p2,p3,p4,
        layout=grid(2,2),
        size=(1000,800)) 
    #savefig(fig,"results/plot"*"s$N"*".png")
end
gif(anim, "anim_fps15.gif", fps = 5);

##
#  X = reshape([i for i in 1:100 for j in 1:100],N,N)
# Y = reshape([j for i in 1:100 for j in 1:100],N,N)
# p1 = surface(X,Y,sol.u[3][:,:,1],title = "[A]",camera=(0,90))
# p2 = surface(X,Y,sol.u[3][:,:,2],title = "[B]",camera=(0,90))
# plot(p1,p2,layout=grid(2,1)) 

