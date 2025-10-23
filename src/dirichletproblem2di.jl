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
    M12::SparseMatrixCSC{Float64, Int64}
    MINV12::SparseMatrixCSC{Float64, Int64}
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
    mg_iters::Array{Int64}
end

function DirichletProb2Di(
    omega::Float64,
    Xmap::Function,
    Ymap::Function,
    qmin::Float64,
    qmax::Float64,
    rmin::Float64,
    rmax::Float64,
    order::Int64,
    ep_tol::Float64;
    Np::Int64 = 5)

    # Use PPW estimate to choose number of gridpoints
    lam = 2*pi/omega
    Nlam = (qmax-qmin)/lam
    PPW = pi*(Nlam/ep_tol)^(1/order)
    Nq = Int(ceil(Nlam*PPW))
    Nlam = (rmax-rmin)/lam
    PPW = pi*(Nlam/ep_tol)^(1/order)
    Nr = Int(ceil(Nlam*PPW))
    println("Number of gridpoints = ",Nq," ",Nr)

    D1Q = derivative_operator(MattssonNordström2004(),
                              1,order,qmin,qmax,Nq+2)
    q_grid = collect(SummationByPartsOperators.grid(D1Q))

    D1R = derivative_operator(MattssonNordström2004(),
                              1,order,rmin,rmax,Nr+2)
    r_grid = collect(SummationByPartsOperators.grid(D1R))

    X = zeros(Nq+2,Nr+2)
    Y = zeros(Nq+2,Nr+2)

    for j = 1:Nr+2
        for i = 1:Nq+2
            X[i,j] = Xmap(q_grid[i],r_grid[j])
            Y[i,j] = Ymap(q_grid[i],r_grid[j])
        end
    end
    N = Nq*Nr

    Xq = zeros(Nq+2,Nr+2)
    Xr = zeros(Nq+2,Nr+2)
    Yq = zeros(Nq+2,Nr+2)
    Yr = zeros(Nq+2,Nr+2)
    Jac = zeros(Nq+2,Nr+2)
    Qx = zeros(Nq+2,Nr+2)
    Rx = zeros(Nq+2,Nr+2)
    Qy = zeros(Nq+2,Nr+2)
    Ry = zeros(Nq+2,Nr+2)

    for j = 1:Nr+2
        Xq[:,j] .= D1Q*X[:,j]
        Yq[:,j] .= D1Q*Y[:,j]
    end
    for i = 1:Nq+2
        Xr[i,:] .= D1R*X[i,:]
        Yr[i,:] .= D1R*Y[i,:]
    end
    Jac .= Xq.*Yr .- Xr.*Yq
    Qx .=  Yr./Jac
    Rx .= -Yq./Jac
    Qy .= -Xr./Jac
    Ry .=  Xq./Jac

    Dr = sparse(D1R)
    Dq = sparse(D1Q)

    Lap2 = spzeros(Nq*Nr,Nq*Nr)
    Lap3 = spzeros(Nq*Nr,Nq*Nr)

    TMP = zeros(Nq+2,Nr+2)
    TMP .= Jac.*(Qx.*Rx .+ Qy.*Ry)
    tmp = reshape(TMP,:)
    DI = spdiagm(tmp)
    Lap3 = kron(Dr,SparseMatrixCSC(I,Nq+2,Nq+2))*DI*kron(SparseMatrixCSC(I,Nr+2,Nr+2),Dq)
    Lap2 = kron(SparseMatrixCSC(I,Nr+2,Nr+2),Dq)*DI*kron(Dr,SparseMatrixCSC(I,Nq+2,Nq+2))

    not_idx = [];
    idx = 1
    for j = 1:Nr+2
        for i = 1:Nq+2
            if i == 1 || i == Nq+2 || j == 1 || j == Nr+2
                push!(not_idx,idx)
            end
            idx = idx+1
        end
    end

    Nidx = collect(1:(Nq+2)*(Nr+2))
    Nidx = Nidx[Not(not_idx)]
    P = spzeros(Float64,Nq*Nr,(Nq+2)*(Nr+2))
    for idx = 1:Nq*Nr
        P[idx,Nidx[idx]] = 1.0
    end
    # Remove zero Dirichlet BC
    Lap3 = P*Lap3*transpose(P)
    Lap2 = P*Lap2*transpose(P)

    Lap1 = spzeros(Nq*Nr,Nq*Nr)
    E = spzeros(Nr,Nr)
    TMP .= Jac.*(Qx.*Qx .+ Qy.*Qy)
    idx = 1
    b = zeros(Nq+2)
    for j = 2:Nr+1
        E[idx,idx] = 1.0
        b .= Vector(TMP[:,j-1])
        D2Q = var_coef_derivative_operator(Mattsson2012(),2,order,
                                           qmin,qmax,Nq+2,abs2)
        D2Q.b .= b
        Dqq = sparse(D2Q)
        Dqq = Dqq[2:end-1,2:end-1]
        Lap1 .= Lap1 .+ kron(E,Dqq)
        E[idx,idx] = 0.0
        idx = idx+1
    end
    E = spzeros(Nq,Nq)
    TMP .= Jac.*(Rx.*Rx .+ Ry.*Ry)
    idx = 1
    b = zeros(Nr+2)
    for i = 2:Nq+1
        E[idx,idx] = 1.0
        b .= Vector(TMP[i-1,:])
        D2R = var_coef_derivative_operator(Mattsson2012(),2,order,
                                           rmin,rmax,Nr+2,abs2)
        D2R.b .= b
        Drr = sparse(D2R)
        Drr = Drr[2:end-1,2:end-1]
        Lap1 .= Lap1 .+ kron(Drr,E)
        E[idx,idx] = 0.0
        idx = idx+1
    end
    b = Vector(TMP[:,1])
    D2Q = var_coef_derivative_operator(Mattsson2012(),2,order,
                                       rmin,rmax,Nq+2,abs2)
    Mq = sparse(mass_matrix(D2Q))
    Mq = Mq[2:end-1,2:end-1]

    b = Vector(TMP[1,:])
    D2R = var_coef_derivative_operator(Mattsson2012(),2,order,
                                       rmin,rmax,Nr+2,abs2)
    Mr = sparse(mass_matrix(D2R))
    Mr = Mr[2:end-1,2:end-1]
    M = kron(Mr,Mq)

    Jinv = reshape(1.0 ./ Jac[2:end-1,2:end-1],Nq*Nr)
    Lap = spdiagm(Jinv)*(Lap1 .+ Lap2 .+ Lap3)
    Mass = spdiagm(reshape(Jac[2:end-1,2:end-1],Nq*Nr))*M

    # Time stepping

    Nt = 10*Np

    dt = sqrt(2.0/cos(2*pi/Nt)-2.0)/omega
    omega = 2.0*pi/Nt/dt
    T = 2.0*pi/omega
    Tp = Np*T

    um = zeros(N)
    u = zeros(N)
    up = zeros(N)
    force = zeros(N)
    rhside = zeros(N)
    # AMG
    IN = sparse(1.0I,N,N)

    hq = q_grid[2]-q_grid[1]
    hr = r_grid[2]-r_grid[1]
    Mass .= Mass / (hq*hr)
    M12 = spdiagm(sqrt.(diag(Mass)))
    MINV12 = spdiagm(1.0 ./ (sqrt.(diag(Mass))))
    G = (IN - 0.5*dt^2*M12*Lap*MINV12)
    ml = ruge_stuben(G,strength = Classical(0.9),
                     presmoother = GaussSeidel(),
                     postsmoother = GaussSeidel(),
                     max_levels = 10,
                     max_coarse = 10)

    precond = aspreconditioner(ml)
    mg_iters = zeros(Int64,2)
    DirichletProb2Di(omega,Nq,Nr,Nt,N,hq,hr,dt,
                     Lap,Mass,M12,MINV12,
                     G,ml,precond,rhside,Np,Tp,T,um,u,up,
                     force,q_grid,r_grid,order,mg_iters)

end
