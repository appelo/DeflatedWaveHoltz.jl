include("waveholtzoperators_exp.jl")

#######################################################################
# Apply the WaveHoltz Operator one time
#######################################################################
function WHI_operator_i!(uproj,uin,DP::Prob2D)
    
    # Use the arrays allocated in the problem struct
    u = DP.u
    um = DP.um
    up = DP.up
    force = DP.force
    rhside = DP.rhside
    dt = DP.dt
    dt2 = dt*dt
    nt = DP.Nt*DP.Np
    A = DP.Lap
    T = DP.Tp

    omega = DP.omega
    cos_omega_dt = cos(omega*dt)
    # Fix up the timestep to be exact at omega
    a0 = 0.5*tan(0.5*omega*dt)/tan(omega*dt)

    # Start from the inital data provided as input
    u .= uin
    # Initialize to have zero velocity
    up .= (2.0*u - dt2*force*cos_omega_dt)
    um, log1 = cg(DP.G,up,Pl=DP.precond,
                  log=true,reltol=1e-14,verbose=false,
                  maxiter=100)
    um .*= 0.5

    # Integration in the fist step
    tt = 0.0
    uproj .= (0.5*(cos(omega*tt)-a0)).*u
    # Loop over time
    for it = 1:nt
        rhside .= (2.0*u .- dt2*force*cos(omega*tt)*cos_omega_dt)
        up, log1 = cg(DP.G,rhside,Pl=DP.precond,
                      log=true,reltol=1e-14,verbose=false,
                      maxiter=100)
        up .-= um
        # Swap
        um .= u
        u .= up
        tt = it*dt
        uproj .+= (cos(omega*(tt))-a0).*u
    end
    # Normalize the integral
    uproj .= (2.0*dt/T).*(uproj-0.5*(cos(omega*T)-a0).*u)
end


#######################################################################
# Apply the WaveHoltz Operator with no forcing one time
#######################################################################
function WHI_operator_homi!(uproj,uin,DP::Prob2D)

    # Use the arrays allocated in the problem struct
    u = DP.u
    um = DP.um
    up = DP.up
    rhside = DP.rhside
    dt = DP.dt
    dt2 = dt*dt
    nt = DP.Nt*DP.Np
    A = DP.Lap
    T = DP.Tp

    omega = DP.omega
    cos_omega_dt = cos(omega*dt)
    # Fix up the timestep to be exact at omega
    a0 = 0.5*tan(0.5*omega*dt)/tan(omega*dt)

    # Start from the inital data provided as input
    u .= uin
    # Initialize to have zero velocity
    up .= 2.0*u
    um, log1 = cg(DP.G,up,Pl=DP.precond,
                  log=true,reltol=1e-14,verbose=false,
                  maxiter=100)
    DP.mg_iters[1] += 1
    DP.mg_iters[2] += log1.iters
    um .*= 0.5

    # Integration in the fist step
    tt = 0.0
    uproj .= (0.5*(cos(omega*tt)-a0)).*u
    # Loop over time
    for it = 1:nt
        rhside .= 2.0*u
        up, log1 = cg(DP.G,rhside,Pl=DP.precond,
                      log=true,reltol=1e-14,verbose=false,
                      maxiter=100)
        DP.mg_iters[1] += 1
        DP.mg_iters[2] += log1.iters
        up .-= um
        # Swap
        um .= u
        u .= up
        tt = it*dt
        uproj .+= (cos(omega*(tt))-a0).*u
    end
    # Normalize the integral
    uproj .= (2.0*dt/T).*(uproj-0.5*(cos(omega*T)-a0).*u)

end

function WHI_sym_operator_homi!(uproj,uin,DP::Prob2D)
    WHI_operator_homi!(uproj,uin,DP)
    uproj .= DP.Mass*uproj
end

function S_WHI_operator_homi!(uproj,uin,DP::Prob2D)
    WHI_operator_homi!(uproj,uin,DP)
    uproj .= uin - uproj
end

function S_WHI_sym_operator_homi!(uproj,uin,DP::Prob2D)
    WHI_operator_homi!(uproj,uin,DP)
    uproj .= uin - uproj
    uproj .= DP.Mass*uproj
end


function WHI_sym_operator_i!(uproj,uin,DP::Prob2D)
    WHI_operator_i!(uproj,uin,DP)
    uproj .= DP.Mass*uproj
end


