using DeflatedWaveHoltz
using Plots

function t()
    order = 2
    ep_tol = 1e-3
    keig = 10
    omega = 5.3

    Np = 10
    
    xmin = -1.0
    xmax =  1.0
    ymin = -2.0
    ymax =  2.0

    DP  =
    DeflatedWaveHoltz.DirichletProb2D(omega,xmin,xmax,ymin,ymax,order,ep_tol,Np
    = Np)

    EVEC1DX = zeros(DP.Nx,DP.Nx)
    EVAL1DX = zeros(DP.Nx)
    for i = 1:DP.Nx
        EVAL1DX[i],EVEC1DX[:,i] = epair_1d_laplace(i,DP.Nx)
    end

    EVEC1DY = zeros(DP.Ny,DP.Ny)
    EVAL1DY = zeros(DP.Ny)
    for i = 1:DP.Ny
        EVAL1DY[i],EVEC1DY[:,i] = epair_1d_laplace(i,DP.Ny)
    end
    
    Lambdas = zeros(DP.N,4)
    ii = 1
    for j = 1:DP.Ny
        for i = 1:DP.Nx
            Lambdas[ii,1] = -(EVAL1DX[i]/DP.hx^2 + EVAL1DY[j]/DP.hy^2)
            Lambdas[ii,2] = i
            Lambdas[ii,3] = j
            Lambdas[ii,4] = 1.0 - bfunex(Lambdas[ii,1],DP)
            ii = ii + 1
        end
    end

    omega_minus_Lambda = copy(Lambdas)
    omega_minus_Lambda[:,4] .= abs.(omega_minus_Lambda[:,4])
    klarr = Int.(omega_minus_Lambda[sortperm(omega_minus_Lambda[:,4]),2:3])
    # Array that holds the eigenvalues to be deflated
    klarr = klarr[1:keig,:]

    eigs = zeros(keig)
    l2a =  zeros(keig)

    for kl = 1:keig
        eig_tmp = -(EVAL1DX[klarr[kl,1]]/DP.hx^2 + EVAL1DY[klarr[kl,2]]/DP.hy^2)
        eigs[kl] = bfunex(eig_tmp,DP)
        l2a[kl] = eig_tmp
    end
    l2 = LinRange(0,3*omega,1000)
    pl = plot(l2,abs.(1.0 .- bfunex.(-l2.^2,DP)),lw=2)
    plot!(pl,sqrt.(-Lambdas[:,1]),abs.(1.0 .- bfunex.(Lambdas[:,1],DP)),marker = :+,seriestype=:scatter)
    plot!(pl,sqrt.(-l2a),abs.(1.0 .- eigs),marker = :c,mc = :black,seriestype=:scatter,ylim = (-0.1,1.6),xlim = (0,3*omega))
    display(pl)
    println("ACR ",maximum(eigs))

    N = DP.N
    Nx = DP.Nx
    Ny = DP.Ny
    W = zeros(N,keig)
    for kl = 1:keig
        for j = 1:Ny
            for i = 1:Nx
                W[i+(j-1)*Nx,kl] = EVEC1DX[i,klarr[kl,1]]*EVEC1DY[j,klarr[kl,2]]
            end
        end
    end

    TMP  = zeros(N,keig)
    WTA  = zeros(keig,N)
    WTAW  = zeros(keig,keig)
    for kl = 1:keig
        eig_tmp = -(EVAL1DX[klarr[kl,1]]/DP.hx^2 + EVAL1DY[klarr[kl,2]]/DP.hy^2)
        eigs[kl] = bfunex(eig_tmp,DP)
        TMP[:,kl] .= eigs[kl]*W[:,kl]
        WTAW[kl,kl] = eigs[kl]
    end
    WTA .= TMP'
    
    return nothing
end

t()
