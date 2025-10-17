function run_example_from_file(omega, xmap::Function, ymap::Function,qmin,qmax,rmin,rmax,order,ep_tol,
                               explicit = false, nev = 5,cgtol = 1e-12, fname = "d.jld2")

    # Set up prolem
    DP = DeflatedWaveHoltz.DirichletProb2Di(omega,xmap,ymap,qmin,qmax,rmin,rmax,order,ep_tol)
    # Set forcing
    set_gauss_forcing!(DP,0.1,0.2)

    DP.force .= DP.M12*DP.force
    S_whi_im!(y,x) = S_WHI_operator_homi!(y,x,DP)
    ALM_WHI = LinearMap(S_whi_im!,DP.N,DP.N,issymmetric = true,ismutating=true,isposdef=true)

    uin = zeros(DP.N)
    b = zeros(DP.N)
    ucg = zeros(DP.N)
    # b = M*Pi[0]
    WHI_operator_i!(b,uin,DP)

    ucg, log1 = cg(ALM_WHI,b,log=true,reltol=cgtol,
                   verbose=true,
                   maxiter=DP.N)
    ucg .= DP.MINV12*ucg
    println("CG MG stats: ", DP.mg_iters[2]/DP.mg_iters[1],"\n")

    # Now do deflated force versions
    # WHI first
    # reset counter
    DP.mg_iters .= 0
    decomp = load(fname,"decomp")
    (ev,Q) = partialeigen(decomp)
    # Project in inner product where eigenvectors are orthogonal
    force = zeros(DP.N)
    force .= DP.force
    coeff = transpose(Q)*force
    DP.force .= force .- Q*coeff

    # reset counter
    DP.mg_iters .= 0
    uwhi = zeros(DP.N)
    uin .= uwhi
    res_whi = zeros(length(log1.data[:resnorm]))
    println("Starting WHI")
    for iter = 1:length(log1.data[:resnorm])
        WHI_operator_i!(uwhi,uin,DP)

        res_whi[iter] = norm(uwhi-uin)
        uin .= uwhi
        println(iter," ",res_whi[iter])
        if(res_whi[iter]/res_whi[1] < cgtol)
            res_whi = res_whi[1:iter]
            break
        end
    end
    # inflate
    for i = 1:nev
        xi = Q[:,i]
        w = omega^2*xi + DP.M12*DP.Lap*DP.MINV12*xi
        cff = 1.0/dot(xi,w)
        uwhi .+= coeff[i]*cff*Q[:,i]
    end
    uwhi .= DP.MINV12*uwhi
    println("Difference = ",norm(ucg-uwhi)/norm(ucg))


    #=
    # Non-ymmetric operator for (1-S) CG
    S_whi_im2!(y,x) = S_WHI_operator_homi!(y,x,DP)
    AGMRES_WHI = LinearMap(S_whi_im2!,DP.N,DP.N,issymmetric = true,ismutating=true,isposdef=true)

    b = zeros(DP.N)
    udefcg = zeros(DP.N)
    uin .= 0.0
    WHI_operator_i!(b,uin,DP)

    udefcg, log_2 = gmres(AGMRES_WHI,b,
    log=true,
    reltol=cgtol,
    restart=50,
    verbose=true,
    maxiter=DP.N)
    println("GMRES + deflate MG stats: ", DP.mg_iters[2]/DP.mg_iters[1])
    for i = 1:nev
    xi = Q[:,i]
    w = (omega^2*DP.Mass + DP.Mass*DP.Lap)*xi
    cff = dot(xi,DP.Mass*xi)/dot(xi,w)
    udefcg .+= coeff[i]*cff*Q[:,i]
    end
    println("Difference = ",norm(ucg-udefcg)/norm(ucg))
    println("--------------------\n\n")
    =#

    b = zeros(DP.N)
    udefcg = zeros(DP.N)
    uin .= 0.0
    WHI_operator_i!(b,uin,DP)
    udefcg, log_2 = cg(ALM_WHI,b,log=true,reltol=cgtol,
                       verbose=true,
                       maxiter=DP.N)
    println("CG + deflate MG stats: ", DP.mg_iters[2]/DP.mg_iters[1])
    for i = 1:nev
        xi = Q[:,i]
        w = omega^2*xi + DP.M12*DP.Lap*DP.MINV12*xi
        cff = 1.0/dot(xi,w)
        udefcg .+= coeff[i]*cff*Q[:,i]
    end
    udefcg .= DP.MINV12*udefcg
    println("Difference = ",norm(ucg-udefcg)/norm(ucg))
    println("--------------------\n\n")
    return log1,log_2,res_whi,DP.x_grid,DP.y_grid,reshape(ucg,DP.Nx,DP.Ny),reshape(udefcg,DP.Nx,DP.Ny),reshape(uwhi,DP.Nx,DP.Ny)
end

function find_deflate(omega, xmap::Function, ymap::Function,qmin,qmax,rmin,rmax,order,ep_tol,
                      explicit = false, nev = 5,evtol = 1e-12, fname = "d.jld2")

    println("Number of ev = ",nev)
    DP = DeflatedWaveHoltz.DirichletProb2Di(omega,xmap,ymap,qmin,qmax,rmin,rmax,order,ep_tol)
    println("Number of gridpoints = ",DP.Nx," ",DP.Ny)
    A_whi_im!(y,x) = WHI_operator_homi!(y,x,DP)
    ALM = LinearMap(A_whi_im!,DP.N,DP.N,issymmetric = true, ismutating=true, isposdef=false)
    decomp, history = partialschur(ALM, nev=nev, tol=evtol, which=:LR)
    jldsave(fname;decomp = decomp)
    println(history)

    return history,DP
end

function compute_DCG_from_file(fname,omega,xmap,ymap,qmin,qmax,rmin,rmax,order,ep_tol,explicit = true,nev = 5, cgtol=1e-12)

    DP = DeflatedWaveHoltz.DirichletProb2Di(omega,xmap,ymap,qmin,qmax,rmin,rmax,order,ep_tol)
    set_gauss_forcing!(DP,0.1,0.2)
    DP.force .= DP.M12*DP.force

    S_whi_im!(y,x) = S_WHI_operator_homi!(y,x,DP)
    ALM_WHI = LinearMap(S_whi_im!,DP.N,DP.N,issymmetric = true,ismutating=true,isposdef=true)
    decomp = load(fname,"decomp")
    (ev,Q) = partialeigen(decomp)

    uin = zeros(DP.N)
    b = zeros(DP.N)
    WHI_operator_i!(b,uin,DP)

    #
    lamd = zeros(nev)
    betd = zeros(nev)
    W = zeros(DP.N,nev)
    TMP  = zeros(DP.N,nev)
    WTA  = zeros(nev,DP.N)
    WTAW  = zeros(nev,nev)
    println("Starting deflation compute")
    for i = 1:nev
        xi = Q[:,i]
        W[:,i] .= xi
        w = DP.Lap*xi
        TMP[:,i] .= ALM_WHI*W[:,i]
        println("case ",i," of ",nev)
    end

    println("DCG preprocess MG stats: ", DP.mg_iters[2]/DP.mg_iters[1],"\n")
    # reset counter
    DP.mg_iters .= 0


    WTA .= TMP'
    WTAW .= WTA*W

    tmp = zeros(DP.N)
    xcg = zeros(DP.N)
    r = zeros(DP.N)
    p = zeros(DP.N)
    q = zeros(DP.N)

    xm = zeros(DP.N)
    xdcg = zeros(DP.N)
    rm = zeros(DP.N)
    mu = zeros(nev)
    rrm = zeros(nev)

    tmp1 = zeros(DP.N)
    tmp2 = zeros(DP.N)


    to = TimerOutput()

    @timeit to "Apply A" tmp .= ALM_WHI*xm

    @timeit to "Vec operation" rm .= b.-tmp

    @timeit to "Apply W^T" rrm .= W'*rm
    @timeit to "Invert W^TAW" rrm .= WTAW\rrm
    @timeit to "Apply W" rm .= W*rrm
    @timeit to "Vec operation" xcg .= xm .+ rm

    @timeit to "Apply A" tmp .= ALM_WHI*xcg
    @timeit to "Vec operation" r .= b .- tmp
    @timeit to "Apply WTA" mu .= WTA*r
    @timeit to "Invert W^TAW" mu .= WTAW\mu
    @timeit to "Apply W" rm .= W*mu
    @timeit to "Vec operation" p .= r .- rm
    @timeit to "Vec operation" rtr = dot(r,r)
    @timeit to "Vec operation" rtr0 = rtr

    max_iter = 500
    history = zeros(max_iter+1)
    history[1] = sqrt(rtr0)
    iter = 1
    while iter < max_iter
        @timeit to "Apply A" q .= ALM_WHI*p
        @timeit to "Vec operation" alpha = rtr/dot(p,q)
        @timeit to "Vec operation" xcg .= xcg .+ alpha*p
        @timeit to "Vec operation" r .= r .- alpha*q
        rtrold = rtr
        @timeit to "Vec operation" rtr = dot(r,r)
        #RDCG = [RDCG sqrt(rtr/rtr0)];
        @timeit to "Vec operation" beta = rtr/rtrold

        @timeit to "Apply WTA" mu .= WTA*r
        @timeit to "Invert W^TAW" mu .= WTAW\mu
        @timeit "Apply W" rm .= W*mu
        @timeit "Vec operation" p .= beta*p .+ r .- rm
        history[iter+1] = sqrt(rtr)
        println(iter," : ",sqrt(rtr/rtr0))

        @timeit to "Vec operation" if(rtr/rtr0 < cgtol^2)
            break
        end
        iter = iter + 1
    end
    println("DCG MG stats: ", DP.mg_iters[2]/DP.mg_iters[1],"\n")
    # reset counter
    DP.mg_iters .= 0
    xcg .= DP.MINV12*xcg
    return xcg,history[1:iter+1]
end

function process_deflation_from_file(fname,DP,svd_tol=1e-12)

    decomp = load(fname,"decomp")
    (ev,Q) = partialeigen(decomp)
    nev = length(ev)
    lamd = zeros(nev)
    betd = zeros(nev)
    lam_plot = collect(LinRange(0,5*DP.omega,1000))
    bet_plot = zeros(1000)
    for  i = 1:1000
        bet_plot[i] = bfun2(lam_plot[i],DP)
    end
    #
    rank_evec = zeros(Int64,nev)
    eig_svd_list = []
    for i = 1:nev

        xi = Q[:,i]
        w = DP.M12*DP.Lap*DP.MINV12*xi
        lam = sqrt(-dot(xi,w))
        lamd[i] = lam
        betd[i] = bfun2(lam,DP)

        Xi = reshape(Q[:,i],DP.Nx,DP.Ny)
        F = svd(Xi)
        energy = reverse(cumsum(reverse(F.S.^2)))
        r = findall(energy .> (svd_tol)^2)
        r = r[end]
        rank_evec[i] = r
        FF = SVD(F.U[:,1:r],F.S[1:r],F.Vt[1:r,:])
        push!(eig_svd_list,FF)
    end
    return rank_evec, eig_svd_list,lamd,betd,lam_plot,bet_plot
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
