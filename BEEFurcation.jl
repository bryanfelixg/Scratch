using Plots, Parameters
using BifurcationKit, Setfield, ForwardDiff, NestedTuples
const BK = BifurcationKit

function pips!(dr,r,p,t)
    @unpack k1,k2,k3,k4,k5,k6,D1,D2 = p
    pi3k,pten,pip3 = r
    pip2 = 1-pip3

    out = similar(r)
    out[1] = k4*(1-pi3k)*pip3 - k3*pi3k*pip2/(D1-pip2)
    out[2] = k1*(1-pip3)*(1-pten) - k2*pten*pip3/(D2-pip3)
    out[3] = k5*(1-pip3)*pi3k - k6*pip3*pten
    out
end

pips(r,p) = pips!(similar(r),r,p,0)
dpips(z,p) = ForwardDiff.jacobian(x->pips(x,p),z)

jet = BK.getJet(pips, dpips)

## options for Krylov-Newton
opt_newton = NewtonPar(tol = 1e-9, maxIter = 100)

# options for continuation
opts_br = ContinuationPar(
    dsmin = 0.0005, dsmax = 0.05, ds = 0.001,
	maxSteps = 20000, 
    nev = 3, 
    newtonOptions = opt_newton,
	# parameter interval
	# pMin = -0.4, pMax = 11.,
	# detect bifurcations with bisection method
	detectBifurcation = 3, nInversion = 4, tolBisectionEigenvalue = 1e-8, dsminBisection = 1e-9)

##

p₀=(k1=1.0 , 
    k2=0.2,#.2, 
    k3=1.0 , 
    k4=0.8 , 
    k5=2. , 
    k6=1., 
    D1=0.0001, 
    D2=.05) ;

p =(k1=1.0 , 
    k2=0.2, 
    k3=1.0 , 
    k4=0.8 , 
    k5=0.2 , 
    k6=0.1, 
    D1=0.0001, 
    D2=.05) ;

##
lens_v = (
    (@lens _.k1),
    (@lens _.k2),
    (@lens _.k3),
    (@lens _.k4),
    (@lens _.k5),
    (@lens _.k6),
    (@lens _.D1),
    (@lens _.D2)
)
##
for ii in 1:length(lens_v)
    
    diagram1 = bifurcationdiagram(jet...,
        # initial point and parameter
        [.01,.01, .9], p,
        # specify the continuation parameter
        lens_v[ii],
        # very important parameter. This specifies the maximum amount of recursion
        # when computing the bifurcation diagram. It means we allow computing branches of branches
        # at most in the present case.
        4,
        (args...) -> setproperties(opts_br; 
            pMin = -1.0, pMax = 10.0, 
            ds = 0.001, dsmax = 0.005, 
            nInversion = 8, 
            detectBifurcation = 3, 
            dsminBisection =1e-18, 
            maxBisectionSteps=20);
        recordFromSolution = (x, p) -> x[3]
    )

    diagram2 = bifurcationdiagram(jet...,
        # initial point and parameter
        [.9,.9, .01], p,
        # specify the continuation parameter
        lens_v[ii],
        # very important parameter. This specifies the maximum amount of recursion
        # when computing the bifurcation diagram. It means we allow computing branches of branches
        # at most in the present case.
        2,
        (args...) -> setproperties(opts_br; 
            pMin = -1.0, pMax = 10.0, 
            ds = 0.001, dsmax = 0.005, 
            nInversion = 8, 
            detectBifurcation = 3, 
            dsminBisection =1e-18, 
            maxBisectionSteps=20);
        recordFromSolution = (x, p) -> x[3]
    )
    p1 = plot(diagram1; 
        ylims=(-0.1,1.01),
        putspecialptlegend=false, 
        markersize=2, 
        plotfold=false, 
        title = "pip3"
    )
    plot!(diagram2; 
        ylims=(-0.1,1.01),
        putspecialptlegend=false, 
        markersize=2, 
        plotfold=false, 
        title = "pip3"
    ) 
    p1|>display
end

# diagram2 = bifurcationdiagram(jet...,
# 	# initial point and parameter
# 	[.9,.9,.9], p,
# 	# specify the continuation parameter
# 	(@lens _.k3),
# 	# very important parameter. This specifies the maximum amount of recursion
# 	# when computing the bifurcation diagram. It means we allow computing branches of branches
# 	# at most in the present case.
# 	2,
# 	(args...) -> setproperties(opts_br; 
#         pMin = -1.0, pMax = 2.0, 
#         ds = 0.001, dsmax = 0.005, 
#         nInversion = 8, 
#         detectBifurcation = 3, 
#         dsminBisection =1e-18, 
#         maxBisectionSteps=20);
# 	recordFromSolution = (x, p) -> x[3])

# You can plot the diagram like

# plot!(diagram2)
