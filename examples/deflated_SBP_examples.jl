using DeflatedWaveHoltz
using LinearAlgebra, SparseArrays, SummationByPartsOperators, IterativeSolvers, AlgebraicMultigrid, LinearMaps, ArnoldiMethod,JLD2, LaTeXStrings,TimerOutputs
using CairoMakie
using Meshes

include("../src/vc_helpers.jl")

function run_case(fname,omega,nev)

    ep_tol = 1e-4
    explicit = false
    ev_tol = 1e-14
    svd_tol=1e-12
    cgtol = 1e-10

    order = 4

    qmin = -1.0
    rmin = -1.0
    qmax =  1.0
    rmax =  1.0

    xmap(q,r) = q + 0.1*sin(2*r)
    ymap(q,r) = r - 0.3*sin(2*q)

    @time DP = DeflatedWaveHoltz.DirichletProb2Di(omega,xmap,ymap,qmin,qmax,rmin,rmax,order,ep_tol,Np=3)

    if 1==2
        history = find_deflate(DP,fname, nev, ev_tol)
    end

    if 1==1
        rank_arr, eig_svd_arr,lamd,betd,lam_plot,bet_plot = process_deflation_from_file(DP,fname,svd_tol)
    end
    
    if 1==1
        
        log1,log_2,res_whi,x,y,ucg,udfcg,uwhi = run_example_from_file(DP,fname,nev,cgtol)

        set_gauss_forcing!(DP,0.1,0.2)
        uHH = (omega^2*1.0I + DP.Lap) \ DP.force
        
        udcg, res_udcg = compute_DCG_from_file(DP,fname,nev,cgtol)
        
        println("ERROR CG on deflated : ",norm(uHH - reshape(udfcg,DP.N)))
        println("ERROR CG : ",norm(uHH - reshape(ucg,DP.N)))
        println("ERROR WHI on deflated: ",norm(uHH - reshape(uwhi,DP.N)))
        println("ERROR = DCG : ",norm(uHH - udcg))

        # Create the figure and axis
        set_theme!(Theme(fontsize = 20,
                         Axis = (titlefontsize = 20, xlabelsize = 20, ylabelsize = 20),
                         Legend = (fontsize = 20,)))
        # Beta function
        fig0 = Figure()
        ax0 = Axis(fig0[1, 1],
                   xlabel = latexstring("\$\\lambda\$"))

        line1 = lines!(ax0, lam_plot,bet_plot, color = :blue, label = latexstring("\$\\beta ( \\tilde{\\lambda},\\tilde{\\omega}, \\tilde{T} )\$"))
        line2 = scatter!(ax0, lamd, betd, color = :black,
                         label=latexstring("deflated(50)\$\$"),
                         marker = :circle, markersize = 10)
        (mm,ii) = findmin(betd)
        line2 = scatter!(ax0, lamd[ii], betd[ii], color = :green,
                         label=latexstring("ACR\$\$"),
                         marker = :circle, markersize = 15)

        axislegend(ax0, position = :rb)
        save(string("beta_",chopsuffix(fname, ".jld2"),".pdf"),fig0)


        fig1 = Figure()
        ax1 = Axis(fig1[1, 1],
                   xlabel = latexstring("Iteration\$\$"),
                   yscale = log10,
                   limits = (0, log1.iters,1e-12,10))
        line1 = lines!(ax1, res_udcg, color = :green,
                       linewidth = 3,
                       label = latexstring("DCG\$\$"))
        line1 = lines!(ax1, log1.data[:resnorm], color = :black,
                       linewidth = 3,
                       label = latexstring("CG\$\$"))

        line1 = lines!(ax1, log_2.data[:resnorm], color = :blue,
                       linestyle = :dash,
                       linewidth = 3,
                       label = latexstring("CG with \$f_{d}\$"))

        line1 = lines!(ax1, res_whi, color = :red,
                       linewidth = 3,
                       label = latexstring("WHI with \$f_{d}\$"))

        nn = collect(1:log1.iters)

        line3 = lines!(ax1, nn, mm.^nn, color = :red,
                       linestyle = :dot,
                       label=latexstring("ACR line \$\$"))

        axislegend(ax1, position = :cb)
        save(string("residual_",chopsuffix(fname, ".jld2"),".pdf"),fig1)

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
        cplot = CairoMakie.contourf!(fig2[1,1], X,Y,reshape(udcg,Nq,Nr),colormap = :hsv,levels = 40)
        Colorbar(fig2[1, 2], cplot)
        save(string("solution_",chopsuffix(fname, ".jld2"),".pdf"),fig2)

        #GG = StructuredGrid(Matrix(X), Matrix(Y))
        #display(viz(GG, showsegments = true))
    end
end


fname = "om10_nev20.jld2"
omega = 5.0
nev = 10
run_case(fname,omega,nev)

#=
fname = "om20_nev50.jld2"
omega = 20.0
nev = 50
run_case(fname,omega,nev)

fname = "om40_nev100.jld2"
omega = 40.0
nev = 100
run_case(fname,omega,nev)
=#
