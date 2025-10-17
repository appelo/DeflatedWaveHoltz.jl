function compute_a0(DP::Prob2D)
    a0 = 0.25-0.25*(tan(pi*DP.Np/DP.Nt))^2
    return a0
end

#######################################################################
# Apply the WaveHoltz Operator one time
#######################################################################
function WHI_operator!(uproj::Array{Float64},uin::Array{Float64},DP::Prob2D)

    # Use the arrays allocated in the problem struct
    u = DP.u
    um = DP.um
    up = DP.up
    force = DP.force
    omega = DP.omega
    dt = DP.dt
    nt = DP.Nt
    A = DP.Lap
    T = DP.Tp
    
    # Fix up the timestep to be exact at omega
    dt2 = (2*sin(dt*omega/2)/omega)^2
    a0 = compute_a0(DP)
    
    # Start from the inital data provided as input
    u .= uin
    
    # Initialize to have zero velocity  
    um .= uin
    mul!(um,A,u,0.5*dt2,1.0)
    um .= um .- (0.5*dt2).*force
    # Integration in the fist step
    tt = 0.0
    uproj .= (1.0*(cos(omega*tt)-a0)).*u
    # Loop over time
    for it = 1:nt-1
        up .= 2.0.*u
        mul!(up,A,u,dt2,1.0)
        up .= up .- (cos(omega*(it-1)*dt)*dt2).*force
        up .-= um
        # Swap
        um .= u
        u .= up
        tt = it*dt
        uproj .+= (cos(omega*(tt))-a0).*u
    end
    # Normalize the integral
    uproj .= (2.0*dt/T).*uproj
end

#######################################################################
# Apply the WaveHoltz Operator with no forcing one time
#######################################################################
function WHI_operator_hom!(uproj,uin,DP::Prob2D)

    # Use the arrays allocated in the problem struct
    u = DP.u
    um = DP.um
    up = DP.up
    omega = DP.omega
    dt = DP.dt
    nt = DP.Nt
    A = DP.Lap
    T = DP.Tp
    
    # Fix up the timestep to be exact at omega
    dt2 = (2*sin(dt*omega/2)/omega)^2
    a0 = compute_a0(DP)
    
    # Start from the inital data provided as input
    u .= uin
    
    # Initialize to have zero velocity  
    um .= uin
    mul!(um,A,u,0.5*dt2,1.0)
    # Integration in the fist step
    tt = 0.0
    uproj .= (1.0*(cos(omega*tt)-a0)).*u
    # Loop over time
    for it = 1:nt-1
        up .= 2.0.*u
        mul!(up,A,u,dt2,1.0)
        up .-= um
        # Swap
        um .= u
        u .= up
        tt = it*dt
        uproj .+= (cos(omega*(tt))-a0).*u
    end
    # Normalize the integral
    uproj .= (2.0*dt/T).*uproj
end

function S_WHI_operator_hom!(uproj,uin,DP::Prob2D)

    # I - Pi
    
    # Use the arrays allocated in the problem struct
    u = DP.u
    um = DP.um
    up = DP.up
    omega = DP.omega
    dt = DP.dt
    nt = DP.Nt
    A = DP.Lap
    T = DP.Tp
    
    # Fix up the timestep to be exact at omega
    dt2 = (2*sin(dt*omega/2)/omega)^2
    a0 = compute_a0(DP)
    
    # Start from the inital data provided as input
    u .= uin
    
    # Initialize to have zero velocity  
    um .= uin
    mul!(um,A,u,0.5*dt2,1.0)
    # Integration in the fist step
    tt = 0.0
    uproj .= (1.0*(cos(omega*tt)-a0)).*u
    # Loop over time
    for it = 1:nt-1
        up .= 2.0.*u
        mul!(up,A,u,dt2,1.0)
        up .-= um
        # Swap
        um .= u
        u .= up
        tt = it*dt
        uproj .+= (cos(omega*(tt))-a0).*u
    end
    # Normalize the integral and subtract from uin
    uproj .= uin - (2.0*dt/T).*uproj
end


