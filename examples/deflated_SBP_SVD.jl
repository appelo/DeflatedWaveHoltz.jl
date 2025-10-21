using DeflatedWaveHoltz
using LinearAlgebra, SparseArrays, SummationByPartsOperators, IterativeSolvers, AlgebraicMultigrid, LinearMaps, ArnoldiMethod,JLD2, LaTeXStrings,TimerOutputs
using CairoMakie
using Meshes

include("../src/vc_helpers.jl")

function run_SVD_case(fname,omega,nev)

    ep_tol = 1e-3
    explicit = false
    ev_tol = 1e-12
    svd_tol=1e-10
    cgtol = 1e-10

    order = 4

    qmin = -1.0
    rmin = -1.0
    qmax =  1.0
    rmax =  1.0

    xmap(q,r) = q + 0.1*sin(2*r)
    ymap(q,r) = r - 0.3*sin(2*q)


    if 1==1
        history,DP = find_deflate(omega,xmap,ymap,qmin,qmax,rmin,rmax,order,ep_tol,explicit,nev,ev_tol,fname)
    else
        DP = DeflatedWaveHoltz.DirichletProb2Di(omega,xmap,ymap,qmin,qmax,rmin,rmax,order,ep_tol)
    end

    rank_arr, eig_svd_arr,lamd,betd,lam_plot,bet_plot = process_deflation_from_file(fname,DP,svd_tol)

    # Create the figure and axis
    set_theme!(Theme(fontsize = 20,
                     Axis = (titlefontsize = 20, xlabelsize = 20, ylabelsize = 20),
                     Legend = (fontsize = 20,)))

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
    

end


fname = "om5_nev100.jld2"
omega = 5.0
nev = 1024
run_SVD_case(fname,omega,nev)


#fname = "om30_nev20.jld2"
#omega = 30.0
#nev = 20
#run_SVD_case(fname,omega,nev)

#fname = "om75_nev100.jld2"
#omega = 75.0
#nev = 200
#run_SVD_case(fname,omega,nev)


