mutable struct DirichletProb2D
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
    function DirichletProb2D(
        omega::Float64,
        xmin::Float64,
        xmax::Float64,
        ymin::Float64,
        ymax::Float64,
        order::Int64,
        ep_tol::Float64;
        Np::Int64 = 1)
        
        # Use PPW estimate to choose number of gridpoints
        lam = 2*pi/omega
        Nlam = (xmax-xmin)/lam
        PPW = pi*(Nlam/ep_tol)^(1/order)
        Nx = Int(ceil(Nlam*PPW))
        Nlam = (ymax-ymin)/lam
        PPW = pi*(Nlam/ep_tol)^(1/order)
        Ny = Int(ceil(Nlam*PPW))
        println(Nx,Ny)
        # SBP operators for zero Dirichlet BC
        D2X = derivative_operator(MattssonNordström2004(),
                                  2,order,xmin,xmax,Nx+2)
        x_grid = grid(D2X)
        x_grid = x_grid[2:end-1]
        Mx = sparse(mass_matrix(D2X))
        Dxx = sparse(D2X)
        Dxx = Dxx[2:end-1,2:end-1]
        Mx = Mx[2:end-1,2:end-1]    

        D2Y = derivative_operator(MattssonNordström2004(),
                                  2,order,ymin,ymax,Ny+2)
        y_grid = grid(D2Y)
        y_grid = y_grid[2:end-1]
        My = sparse(mass_matrix(D2Y))
        Dyy = sparse(D2Y)
        Dyy = Dyy[2:end-1,2:end-1]
        My = My[2:end-1,2:end-1]    

        Mass = kron(My,Mx)
        Lap = kron(Dyy,sparse(I, Nx, Nx)) + kron(sparse(I, Ny, Ny),Dxx)
        N = Nx*Ny
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
        new(omega,Nx,Ny,Nt,N,hx,hy,dt,
            Lap,Mass,Np,Tp,T,um,u,up,
            force,x_grid,y_grid,order)
    end
    function DirichletProb2D(
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
        println(Nx,Ny)
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
        new(omega,Nx,Ny,Nt,N,hx,hy,dt,
            Lap,Mass,Np,Tp,T,um,u,up,
            force,x_grid,y_grid,order)
    end

end
