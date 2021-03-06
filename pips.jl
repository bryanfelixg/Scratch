#line
using DiffEqOperators
using SparseArrays
using LinearAlgebra
using Plots
using Sundials
using DifferentialEquations
#using BenchmarkTools
#using ProfileView
##
x0 = 0.
x1 = 2.
N = 100

dx = (x1-x0)/(N-1)
x = x0:dx:x1 
tspan=(0.,100.)

#p  = (1.0 , .2, .5 , 0.8 , 2. , 1., 0.0001, .05) ;

p  = (1.0 , .7, .4 , 0.8 , 2. , 1., 0.3, 0.1, 0.0001, .05) ;


#gr()
#plot(x[2:end-1],u)


## Operators
Δx = CenteredDifference{1}(2,2,dx,N,1) ;
Δy = CenteredDifference{2}(2,2,dx,N,1) ;
bcx = Neumann0BC(dx,1); # left, right
bcy = Neumann0BC(dx,1); # down, up

Axx = Array(Δx*bcx)[1];
Ayy = Array(Δy*bcy)[1] ;

# GeneralBC([1.,1.],[.5,1.],dx,1) # α1 + α2*u + α3*u'... = 0
# RobinBC((0.,1.,0.),(0.,1.,0.),dx,1)
# Neumann0BC(dx,1)
# PeriodicBC(eltype(u)) 
# Dirichlet0BC(eltype(u))


## Problem 

dx_pi3k = zeros(N,N)
dy_pi3k = zeros(N,N)
Δpi3k   = zeros(N,N)

dx_pten = zeros(N,N)
dy_pten = zeros(N,N)
Δpten  = zeros(N,N)

function basic!(dr,r,p,t)
    k1,k2,k3,k4,k5,k6,M1,M2,D1,D2,n = p
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
    # dy_pi3k= Δy*(bcy*pi3k')'
    # dx_pi3k= Δx*bcx*pi3k
    # dx_pten= Δx*bcx*pten
    # dy_pten= Δy*(bcy*pten')'
    @. Δpi3k = dx_pi3k+dy_pi3k
    @. Δpten = dx_pten+dy_pten #Δx*bcx*pten+ (Δy*bcy)*pten
    #@. Δpi3k = dx_pi3k + dy_pi3k #Δx*bcx*A + Δy*(bcy*A')'
    #@. Δpten = dx_pten + dy_pten #Δx*bcx*A + Δy*(bcy*A')'
    @. d_pi3k .= D1*Δpi3k + k4*(1-pi3k)*pip3 - k3*pi3k*(1-pip3)/(M2 + (1-pip3))
    @. d_pten .= D2*Δpten + k1*(1-pip3)*(1-pten) - k2*pten*pip3/(M1 + pip3)
    @. d_pip3 .= k5*(1-pip3)*pi3k - k6*pip3*pten
end

## Initial Condition

r0 = zeros(N,N,3);
r0[:,:,1] .= 0. # rand.();
r0[:,:,2] .= .999;
r0[:,:,3] .= 0. #rand.();

r0[50:50,50:50,1] .= 0.009

##


prob = ODEProblem(basic!,r0,tspan,p);
##

sol =solve(
    prob,
    CVODE_BDF(linear_solver=:GMRES),
    save_everystep=false,
    saveat = range(tspan[1],tspan[2], length=100)
    ) ;

sol.retcode|>println
#SparseMatrixCSC(Δ)

##

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
    #savefig(fig,"results/plot"*"d1=$(p[end-1])_d2=$(p[end])"*".png")
end
##
gif(anim, "anim_fps15.gif", fps = 15); 