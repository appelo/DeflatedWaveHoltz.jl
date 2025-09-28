using DeflatedWaveHoltz
using LinearAlgebra, SparseArrays, SummationByPartsOperators, IterativeSolvers, AlgebraicMultigrid, LinearMaps, ArnoldiMethod, Plots,JLD2,LaTeXStrings

function construct_grid(x_left,x_right,y_left,y_right,Nx,Ny)
    dx = (x_right-x_left)/(Nx+1)
    dy = (y_right-y_left)/(Ny+1)

    x_grid = zeros(Nx+2);
    y_grid = zeros(Ny+2);
    for i = 0:Nx+1
        x_grid[i+1] = x_left+i*dx;
    end

    for j = 0:Ny+1
        y_grid[j+1] = y_left+j*dy;
    end

    return x_grid,y_grid,dx,dy
end

function c_square(x,y)
    res = 1.0 - 0.5/(1.0+(x+0.1)^2+(y-0.2)^2)^4
    return res;
end

function spatial_discretization(x_interior,y_interior,dx,dy,Nx,Ny,N)
    # Definte the index matrix
    ind_matrix = zeros(Nx,Ny);
    ind_val = 1;
    for i = 1:Nx
        for j = 1:Ny
            ind_matrix[i,j] = ind_val;
            ind_val += 1;
        end
    end
    # Assemble spatial discretization
    row_array = zeros(5*N);
    col_array = zeros(5*N);
    val_array = zeros(5*N);
    ind = 1;

    for i = 1:Nx
        xx = x_interior[i];
        for j = 1:Ny
            yy = y_interior[j];
            #vel = c_square(xx,yy)^2;
            #println(c_square(xx,yy))

            # Center
            row_array[ind] = ind_matrix[i,j];
            col_array[ind] = ind_matrix[i,j];
            val_array[ind] = -( c_square(xx-0.5*dx,yy)+c_square(xx+0.5*dx,yy) )/dx^2-
                ( c_square(xx,yy-0.5*dy)+c_square(xx,yy+0.5*dy) )/dy^2

            ind = ind+1;

            # Left
            if (i>1)
                row_array[ind] = ind_matrix[i,j];
                col_array[ind] = ind_matrix[i-1,j];
                val_array[ind] = c_square(xx-0.5*dx,yy)/dx^2;
                ind = ind+1;
            end

            # Right
            if (i<Nx)
                row_array[ind] = ind_matrix[i,j];
                col_array[ind] = ind_matrix[i+1,j];
                val_array[ind] = c_square(xx+0.5*dx,yy)/dx^2;
                ind = ind+1;
            end

            # Top
            if (j<Ny)
                row_array[ind] = ind_matrix[i,j];
                col_array[ind] = ind_matrix[i,j+1];
                val_array[ind] = c_square(xx,yy+0.5*dy)/dy^2;
                ind = ind+1;
            end

            # Bottom
            if (j>1)
                row_array[ind] = ind_matrix[i,j];
                col_array[ind] = ind_matrix[i,j-1];
                val_array[ind] = c_square(xx,yy-0.5*dy)/dy^2;
                ind = ind+1;
            end

        end
    end
    A = sparse(row_array[1:ind-1],col_array[1:ind-1],val_array[1:ind-1]);
    return A
end


function find_deflate(omega, ep_tol, explicit = true, nev = 5,evtol = 1e-12, fname = "d.jld2")

    println("Number of ev = ",nev)
    # Use PPW estimate to choose number of gridpoints
    lam = 2*pi/omega
    Nlam = 2.0/lam
    PPW = pi*(Nlam/ep_tol)^(1/2)
    Nx = Int(ceil(Nlam*PPW))
    Ny = Nx
    N = Nx*Ny
    println("(Nx,Ny,N) = (",Nx,", ",Ny,", ",N,")")
    x_grid,y_grid,dx,dy = construct_grid(0.0,pi,0.0,pi,Nx,Ny)
    x_interior = x_grid[2:Nx+1];
    y_interior = y_grid[2:Ny+1];
    Lap = spatial_discretization(x_interior,y_interior,dx,dy,Nx,Ny,N)

    if explicit
        DP = DeflatedWaveHoltz.DirichletProb2D(omega,x_interior,y_interior,Lap,ep_tol)
        A_whi_ex!(y,x) = WHI_operator_hom!(y,x,DP)
        ALM = LinearMap(A_whi_ex!,DP.N,DP.N,issymmetric = true,ismutating=true,isposdef=false)
    else
        DP = DeflatedWaveHoltz.DirichletProb2Di(omega,x_interior,y_interior,Lap,ep_tol)
        A_whi_im!(y,x) = WHI_operator_homi!(y,x,DP)
        ALM = LinearMap(A_whi_im!,DP.N,DP.N,issymmetric = true,ismutating=true,isposdef=false)
    end
    decomp, history = partialschur(ALM, nev=nev, tol=evtol, which=:LR)
    jldsave(fname;decomp = decomp)
    println(history)
    return history
end

function run_example_from_file(omega, ep_tol, explicit = true, nev = 5,cgtol = 1e-12,fname="d.jld2")
    
    # Use PPW estimate to choose number of gridpoints
    lam = 2*pi/omega
    Nlam = 2.0/lam
    PPW = pi*(Nlam/ep_tol)^(1/2)
    Nx = Int(ceil(Nlam*PPW))
    Ny = Nx
    N = Nx*Ny
    
    x_grid,y_grid,dx,dy = construct_grid(0.0,pi,0.0,pi,Nx,Ny)
    x_interior = x_grid[2:Nx+1];
    y_interior = y_grid[2:Ny+1];
    Lap = spatial_discretization(x_interior,y_interior,dx,dy,Nx,Ny,N)

    if explicit
        DP = DeflatedWaveHoltz.DirichletProb2D(omega,x_interior,y_interior,Lap,ep_tol)
    else
        DP = DeflatedWaveHoltz.DirichletProb2Di(omega,x_interior,y_interior,Lap,ep_tol)
    end
    
    set_gauss_forcing!(DP,0.1,0.2)
    uin = zeros(DP.N)
    uproj = zeros(DP.N)
    force = zeros(DP.N)
    force .= DP.force

    if explicit
        S_whi_ex!(y,x) = S_WHI_operator_hom!(y,x,DP)
        ALM_WHI = LinearMap(S_whi_ex!,DP.N,DP.N,issymmetric = true,ismutating=true,isposdef=true)
    else
        S_whi_im!(y,x) = S_WHI_operator_homi!(y,x,DP)
        ALM_WHI = LinearMap(S_whi_im!,DP.N,DP.N,issymmetric = true,ismutating=true,isposdef=true)
    end

    b = zeros(DP.N)
    ucg = zeros(DP.N)
    
    if explicit
        WHI_operator!(b,uin,DP)
    else
        WHI_operator_i!(b,uin,DP) 
    end
    ucg, log1 = cg(ALM_WHI,b,log=true,reltol=cgtol,
                   verbose=true,
                   maxiter=DP.N)
    
    if explicit
        A_whi_ex!(y,x) = WHI_operator_hom!(y,x,DP)
        ALM = LinearMap(A_whi_ex!,DP.N,DP.N,issymmetric = true,ismutating=true,isposdef=false)
    else
        A_whi_im!(y,x) = WHI_operator_homi!(y,x,DP)
        ALM = LinearMap(A_whi_im!,DP.N,DP.N,issymmetric = true,ismutating=true,isposdef=false)
    end
    decomp = load(fname,"decomp")
    # deflate force
    lamd = zeros(nev)
    betd = zeros(nev)
    for i = 1:nev
        DP.force .-= dot(force,decomp.Q[:,i])*decomp.Q[:,i]
        xi = decomp.Q[:,i]
        w = Lap*xi
        lam = sqrt(-dot(xi,w))/dot(xi,xi)
        lamd[i] = lam
        betd[i] = bfun2(lam,DP)
    end

    b = zeros(DP.N)
    udefcg = zeros(DP.N)

    if explicit
        WHI_operator!(b,uin,DP)
    else
        WHI_operator_i!(b,uin,DP) 
    end
    udefcg, log2 = cg(ALM_WHI,b,log=true,reltol=cgtol,
                      verbose=true,
                      maxiter=DP.N)
    for i = 1:nev
        xi = decomp.Q[:,i]
        w = Lap*xi
        lam2 = -dot(xi,w)/dot(xi,xi)
        udefcg .+= dot(force,decomp.Q[:,i])/(omega^2-lam2)*decomp.Q[:,i]
    end
    pl1 = contour(DP.x_grid,DP.y_grid,transpose(reshape(ucg,DP.Nx,DP.Ny)),
                  aspect_ratio = 1.0)
    pl2 = contour(DP.x_grid,DP.y_grid,transpose(reshape(udefcg,DP.Nx,DP.Ny)),
                  aspect_ratio = 1.0)
    println("Difference = ",norm(ucg-udefcg)/norm(udefcg))
    
    return pl1,pl2,log1,log2,DP.x_grid,DP.y_grid,reshape(ucg,DP.Nx,DP.Ny),reshape(udefcg,DP.Nx,DP.Ny)
end

function process_deflation_from_file(fname,omega,ep_tol,explicit = true,nev = 5, svd_tol=1e-12)
    
    # Use PPW estimate to choose number of gridpoints
    lam = 2*pi/omega
    Nlam = 2.0/lam
    PPW = pi*(Nlam/ep_tol)^(1/2)
    Nx = Int(ceil(Nlam*PPW))
    Ny = Nx
    N = Nx*Ny
    
    x_grid,y_grid,dx,dy = construct_grid(0.0,pi,0.0,pi,Nx,Ny)
    x_interior = x_grid[2:Nx+1];
    y_interior = y_grid[2:Ny+1];
    Lap = spatial_discretization(x_interior,y_interior,dx,dy,Nx,Ny,N)

    if explicit
        DP = DeflatedWaveHoltz.DirichletProb2D(omega,x_interior,y_interior,Lap,ep_tol)
    else
        DP = DeflatedWaveHoltz.DirichletProb2Di(omega,x_interior,y_interior,Lap,ep_tol)
    end
    if explicit
        S_whi_ex!(y,x) = S_WHI_operator_hom!(y,x,DP)
        ALM_WHI = LinearMap(S_whi_ex!,DP.N,DP.N,issymmetric = true,ismutating=true,isposdef=true)
    else
        S_whi_im!(y,x) = S_WHI_operator_homi!(y,x,DP)
        ALM_WHI = LinearMap(S_whi_im!,DP.N,DP.N,issymmetric = true,ismutating=true,isposdef=true)
    end
    if explicit
        A_whi_ex!(y,x) = WHI_operator_hom!(y,x,DP)
        ALM = LinearMap(A_whi_ex!,DP.N,DP.N,issymmetric = true,ismutating=true,isposdef=false)
    else
        A_whi_im!(y,x) = WHI_operator_homi!(y,x,DP)
        ALM = LinearMap(A_whi_im!,DP.N,DP.N,issymmetric = true,ismutating=true,isposdef=false)
    end
    decomp = load(fname,"decomp")

    # 
    lamd = zeros(nev)
    betd = zeros(nev)
    for i = 1:nev
        xi = decomp.Q[:,i]
        w = Lap*xi
        lam = sqrt(-dot(xi,w))/dot(xi,xi)
        lamd[i] = lam
        betd[i] = bfun2(lam,DP)
    end

    lam_plot = collect(LinRange(0,5*omega,1000))
    bet_plot = zeros(1000)
    for i = 1:1000
        bet_plot[i] = bfun2(lam_plot[i],DP)
    end
    rank_evec = zeros(Int64,nev)
    for i = 1:nev
        Xi = reshape(decomp.Q[:,i],DP.Nx,DP.Ny)
        F = svd(Xi)
        energy = reverse(cumsum(reverse(F.S.^2)))
        r = findall(energy .> (svd_tol)^2)
        r = r[end]
        rank_evec[i] = r
    end
    return lamd,betd,lam_plot,bet_plot,rank_evec,DP.Nx,DP.Ny
end


function bfun(lam,DP)
    dt = DP.dt
    omega = DP.omega
    T = DP.T
    alpha = tan(0.5*omega*dt)/tan(omega*dt)
    sinc_d(z,S,dt) = sin(z*S)/(T*tan(dt*z/2.0)/(dt/2.0))
    return sinc_d(omega+lam,T,dt) + sinc_d(omega-lam,T,dt) - alpha*sinc_d(lam,T,dt)
end

function bfun2(lam,DP)

    # Use the arrays allocated in the problem struct
    u = 1.0
    um = 0.0
    up = 0.0
    dt = DP.dt
    dt2 = dt*dt
    nt = DP.Nt
    A = lam^2
    T = DP.Tp
    omega = DP.omega
    # Fix up the timestep to be exact at omega
    a0 = 0.5*tan(0.5*omega*dt)/tan(omega*dt)
    # Initialize to have zero velocity  
    um = 0.5/(1.0 + 0.5*(dt*lam)^2)*2.0*u
    # Integration in the fist step
    tt = 0.0
    uproj = (0.5*(cos(omega*tt)-a0))*u
    # Loop over time
    for it = 1:nt
        up = 2.0*u/(1.0 + 0.5*(dt*lam)^2) - um
        # Swap
        um = u
        u = up
        tt = it*dt
        uproj += (cos(omega*(tt))-a0)*u
    end
    # Normalize the integral
    uproj = (2.0*dt/T)*(uproj-0.5*(cos(omega*T)-a0)*u)
    return uproj
end

fname = "om30_nev100.jld2"
omega = 30.0
nev = 100
ep_tol = 1e-3
explicit = false
ev_tol = 1e-12
svd_tol=1e-12
cgtol = 1e-12

history = find_deflate(omega, ep_tol, explicit, nev, ev_tol, fname)
lamd,betd,lam_plot,bet_plot,rank_evec,Nx,Ny = process_deflation_from_file(fname,omega,ep_tol,explicit,nev,svd_tol)

default(titlefont = (20, "times"), legendfontsize = 12, tickfont = 12, guidefont=14)
pl0 = plot(lam_plot,bet_plot,lw=2,label=L"\beta_{\rm d}(\lambda)",xlabel=L"\lambda")
plot!(pl0,lamd,betd,marker = :circle, seriestype=:scatter, label=L"\beta_{\rm d}(\lambda_k)",ylims=(-0.6,1.1),yticks = -0.5:0.25:1.0)
savefig(pl0,string(chopsuffix(fname, ".jld2"),".pdf"))

pl1,pl2,log1,log2,x,y,ucg,udfcg = run_example_from_file(omega,ep_tol,explicit,nev,cgtol,fname)

pl3 = plot(log1.data[:resnorm],lw=2,label=L"\textrm{CG}",xlabel="Iteration",yscale=:log10)
plot!(pl3,log2.data[:resnorm],lw=2,label=L"\textrm{CG \, on } \, f_d")
