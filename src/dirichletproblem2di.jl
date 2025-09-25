mutable struct DirichletProb2Di{T1,T2}
    omega::Float64
    # Discretization parameters
    Nx::Int64
    Ny::Int64
    Nt::Int64
    N::Int64
    hx::Float64
    hy::Float64
    dt::Float64
    Lap::SparseMatrixCSC{Float64, Int64}
    Mass::SparseMatrixCSC{Float64, Int64}
    G       ::SparseMatrixCSC{Float64, Int64}
    ml      ::T1
    precond ::T2
    rhside::Array{Float64}
    Np::Int64
    Tp::Float64
    T::Float64
    um::Array{Float64}
    u::Array{Float64}
    up::Array{Float64}
    force::Array{Float64}
    x_grid::Array{Float64}
    y_grid::Array{Float64}
    order::Int64
#=
    function DirichletProb2Di(
        omega::Float64,
        x_grid::Array{Float64},
        y_grid::Array{Float64},
        Lap::SparseMatrixCSC{Float64, Int64},
        ep_tol::Float64;
        Np::Int64 = 1)

        # assume order is 2
        order = 2
        Nx = length(x_grid)
        Ny = length(x_grid)
        N = Nx*Ny

        Mass = kron(sparse(I, Nx, Nx),sparse(I, Ny, Ny))
        hx = x_grid[2]-x_grid[1]
        hy = y_grid[2]-y_grid[1]

        # Time stepping 
        T = 2.0*pi/omega
        Tp = Np*T
        dt = 0.5*min(hx,hy)
        Nt = Np*max(Int(ceil(T/dt)),5)
        dt = Tp/Nt
        um = zeros(N)
        u = zeros(N)
        up = zeros(N)
        force = zeros(N)

        # AMG
        IN = sparse(I,N,N)
        G = IN - 0.5*dt*dt.*Lap
        ml = ruge_stuben(G,strength = Classical(0.9),
                         presmoother = GaussSeidel(),
                         postsmoother = GaussSeidel(),
                         max_levels = 10,
                         max_coarse = 10)
        
        precond = aspreconditioner(ml)
        new{typeof(ml),typeof(precond)}(omega,Nx,Ny,Nt,N,hx,hy,dt,
            Lap,Mass,G,ml,precond,Np,Tp,T,um,u,up,
            force,x_grid,y_grid,order)
    end
=#
end

function DirichletProb2Di(
    omega::Float64,
    x_grid::Array{Float64},
    y_grid::Array{Float64},
    Lap::SparseMatrixCSC{Float64, Int64},
    ep_tol::Float64;
    Np::Int64 = 1)
    
    # assume order is 2
    order = 2
    Nx = length(x_grid)
    Ny = length(x_grid)
    N = Nx*Ny
    
    hx = x_grid[2]-x_grid[1]
    hy = y_grid[2]-y_grid[1]
    
    # Time stepping 
    
    Nt = 10 #Np*max(Int(ceil(T/dt)),5)

    dt = sqrt(2.0/cos(2*pi/Nt)-2.0)/omega
    omega = 2.0*pi/Nt/dt
    T = 2.0*pi/omega
    Tp = Np*T
    
    #    dt = Tp/Nt
    #    T = 2.0*pi/omega
    #    Tp = Np*T
    #    dt = 0.5*min(hx,hy)
    
    um = zeros(N)
    u = zeros(N)
    up = zeros(N)
    force = zeros(N)
    rhside = zeros(N)
    # AMG
    IN = sparse(1.0I,N,N)
    Mass = sparse(1.0I,N,N)
    
    G = IN - 0.5*dt^2*Lap
    ml = ruge_stuben(G,strength = Classical(0.9),
                     presmoother = GaussSeidel(),
                     postsmoother = GaussSeidel(),
                     max_levels = 10,
                     max_coarse = 10)
    
    precond = aspreconditioner(ml)
    DirichletProb2Di(omega,Nx,Ny,Nt,N,hx,hy,dt,
                     Lap,Mass,G,ml,precond,rhside,Np,Tp,T,um,u,up,
                     force,x_grid,y_grid,order)
end


