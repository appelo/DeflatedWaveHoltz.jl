using DeflatedWaveHoltz
using LinearAlgebra, SparseArrays, SummationByPartsOperators, IterativeSolvers, AlgebraicMultigrid, LinearMaps, ArnoldiMethod,JLD2, LaTeXStrings,TimerOutputs
using CairoMakie
using Meshes

include("../src/vc_helpers.jl")

function run_SVD_case(fname,omega,nev)

    to = TimerOutput()
    ep_tol = 1e-4
    explicit = false
    ev_tol = 1e-14
    svd_tol=1e-7
    cgtol = 1e-7

    order = 4

    qmin = -1.0
    rmin = -1.0
    qmax =  1.0
    rmax =  1.0

    xmap(q,r) = q + 0.1*sin(2*r)
    ymap(q,r) = r - 0.3*sin(2*q)

    @timeit to "Set up problem" DP = DeflatedWaveHoltz.DirichletProb2Di(omega,xmap,ymap,qmin,qmax,rmin,rmax,order,ep_tol,Np=3)
    if 1==2
        history = find_deflate(DP,fname, nev, ev_tol)
    end

    @timeit to "Load eigs" decomp = load(fname,"decomp")
    @timeit to "Partial Eigen" (ev,Q) = partialeigen(decomp)

    @timeit to "Compute sol EIG" log1,res_whi,udef_force,uwhi = compare_deflated_example_from_file(DP,Q,fname,nev,cgtol)
    @timeit to "Compress into SVD" rank_arr, eig_svd_arr,lamd,betd,lam_plot,bet_plot = process_deflation_from_file(DP,fname,svd_tol)
    
    compress_ratio = 0 
    for i = 1:nev
        QSVD = eig_svd_arr[i].U*diagm(eig_svd_arr[i].S)*eig_svd_arr[i].Vt
        compress_ratio += length(eig_svd_arr[i].U)+length(eig_svd_arr[i].S)+length(eig_svd_arr[i].Vt)
        Q[:,i] .= reshape(QSVD,:)
    end
    
    @timeit to "Compute sol SVD" log1SVD,res_whiSVD,udef_forceSVD,uwhiSVD = compare_deflated_example_from_file(DP,Q,fname,nev,cgtol)
    show(to)
    println("\n\n")
    
    compress_ratio /= (nev*DP.N)
    println("Compress_ratio : ",compress_ratio)

    println("Diff CG ",norm(udef_forceSVD .- udef_force))
    println("Diff CG ",norm(uwhiSVD .- uwhi))
    # Create the figure and axis
    set_theme!(Theme(fontsize = 20,
                     Axis = (titlefontsize = 20, xlabelsize = 20, ylabelsize = 20),
                     Legend = (fontsize = 20,)))

    fig7 = Figure()
    ax7 = Axis(fig7[1, 1],
               xlabel = latexstring("Iteration\$\$"),
               yscale = log10,
               limits = (0, max(log1.iters,log1SVD.iters),1e-12,10))
    line7 = lines!(ax7, log1.data[:resnorm], color = :black,
                   linewidth = 3,
                   label = latexstring("CG EIG \$\$"))
    line7 = lines!(ax7, log1SVD.data[:resnorm], color = :black,
                   linestyle = :dot,
                   linewidth = 3,
                   label = latexstring("CG SVD \$\$"))
    line7 = lines!(ax7, res_whi, color = :blue,
                   linewidth = 3,
                   label = latexstring("WH EIG \$\$"))
    line7 = lines!(ax7, res_whiSVD, color = :blue,
                   linestyle = :dot,
                   linewidth = 3,
                   label = latexstring("WH SVD \$\$"))
    axislegend(ax7, position = :lb)
    save(string("EIGvsSVD_residual_",chopsuffix(fname, ".jld2"),".pdf"),fig7)
   


    # Rank as a function of ev
    fig0 = Figure()
    ax0 = Axis(fig0[1, 1])
    
    line1 = scatter!(ax0, rank_arr, color = :blue,
                     label=latexstring("rank of SVD\$ \$"),
                     marker = :circle, markersize = 15)
    axislegend(ax0, position = :rb)
    save(string("svd_",chopsuffix(fname, ".jld2"),".pdf"),fig0)

    # Beta function
    fig1 = Figure()
    ax1 = Axis(fig1[1, 1],
               xlabel = latexstring("\$\\lambda\$"))
    
    line1 = lines!(ax1, lam_plot,bet_plot, color = :blue, label = latexstring("\$\\beta ( \\tilde{\\lambda},\\tilde{\\omega}, \\tilde{T} )\$"))
    line2 = scatter!(ax1, lamd, betd, color = :black,
                     label=latexstring("deflated(50)\$\$"),
                     marker = :circle, markersize = 10)
    (mm,ii) = findmin(betd)
    #line2 = scatter!(ax1, lamd[ii], betd[ii], color = :green,
    #                 label=latexstring("ACR\$\$"),
    #                 marker = :circle, markersize = 15)
    
    axislegend(ax1, position = :rb)
    save(string("beta_",chopsuffix(fname, ".jld2"),".pdf"),fig1)

    
    Nq = DP.Nx
    Nr = DP.Ny
    X = zeros(Nq,Nr)
    Y = zeros(Nq,Nr)
    
    for j = 2:Nr+1
        for i = 2:Nq+1
            X[i-1,j-1] = xmap(DP.x_grid[i],DP.y_grid[j])
            Y[i-1,j-1] = ymap(DP.x_grid[i],DP.y_grid[j])
        end
    end
    

    fig2 = Figure()
    Axis(fig2[1, 1],
         xlabel = latexstring("\$x\$-axis"),
         ylabel = latexstring("\$y\$-axis"),
         aspect = DataAspect())
    uplot = eig_svd_arr[1].U*diagm(eig_svd_arr[1].S)*eig_svd_arr[1].Vt
    cplot = CairoMakie.contourf!(fig2[1,1], X,Y,reshape(uplot,Nq,Nr),colormap = :hsv,levels = 40)
    Colorbar(fig2[1, 2], cplot)
    save(string("svd_cont_",chopsuffix(fname, ".jld2"),".pdf"),fig2)

    
    
    return nothing
end


#fname = "om5_nev10.jld2"
#omega = 5.0
#nev = 10
#run_SVD_case(fname,omega,nev)

#fname = "om20_nev50.jld2"
#omega = 20.0
#nev = 50
#run_SVD_case(fname,omega,nev)

fname = "om40_nev100.jld2"
omega = 40.0
nev = 100
run_SVD_case(fname,omega,nev)

