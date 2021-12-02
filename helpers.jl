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