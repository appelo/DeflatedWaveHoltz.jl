
function epair_1d_laplace(i,n)
    lam = 2*(1-cos(i*pi/(n+1)))
    evec = sqrt(2/(n+1))*sin.(pi/(n+1)*i*collect(1:n))
    return lam,evec
end

# compute the beta function in the eigenvalues of 
# the WaveHoltz operator 
function bfunex(lam2::Float64,DP::Prob2D)

    omega = DP.omega
    dt = DP.dt
    nt = DP.Nt
    T = DP.Tp
    # Fix up the timestep to be exact at omega
    dt2 = (2*sin(dt*omega/2)/omega)^2
    a0 = compute_a0(DP)
    # initialization 
    uproj = 0.0
    u = 1.0 # come from the Identity operator
    # Initilize u^{-1}, this results in the half term
    um = u+0.5*dt2*lam2*u 
    tt = 0.0
    uproj = (1.0*(cos(omega*tt)-a0))*u
    for it = 1:nt-1
        up = 2*u-um+dt2*lam2*u
        um = u
        u  = up
        tt = it*dt
        uproj += (cos(omega*(tt))-a0)*u
    end
    #tt = T
    uproj = (2.0*dt/T)*uproj 
    return uproj
end
