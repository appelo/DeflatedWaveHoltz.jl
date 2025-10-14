using DeflatedWaveHoltz
using LinearAlgebra, SparseArrays, SummationByPartsOperators, IterativeSolvers, AlgebraicMultigrid, LinearMaps, ArnoldiMethod,JLD2, LaTeXStrings,TimerOutputs
using CairoMakie

function c_square(x,y)
    res = 1.0 - 0.5/(1.0+(x+0.1)^2+(y-0.2)^2)^4
    return res;
end


include("../src/vc_helpers.jl")

#fname = "om10_nev50.jld2" # 50 of 50 eigenvalues in 325 matrix-vector products
#fname = "om20_nev50.jld2" # 50 of 50 eigenvalues in 1431 matrix-vector products
fname = "om30_nev50.jld2" # 50 of 50 eigenvalues in 3551 matrix-vector products
omega = 30.0
nev = 50
ep_tol = 1e-2
explicit = false
ev_tol = 1e-12
svd_tol=1e-12
cgtol = 1e-10

# history = find_deflate(omega, ep_tol, explicit, nev, ev_tol, fname)
lamd,betd,lam_plot,bet_plot,rank_evec,Nx,Ny = process_deflation_from_file(fname,omega,ep_tol,explicit,nev,svd_tol)

log1,log2,res_whi,x,y,ucg,udfcg,uwhi = run_example_from_file(omega,ep_tol,explicit,nev,cgtol,fname)
udcg, res_udcg = compute_DCG_from_file(fname,omega,ep_tol,explicit,nev,cgtol)



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
           yscale = log10)
line1 = lines!(ax1, res_whi, color = :red,
               linewidth = 3,
               label = latexstring("WHI with \$f_{d}\$"))
line1 = lines!(ax1, log1.data[:resnorm], color = :blue,
               linewidth = 3,
               label = latexstring("CG \$\$"))

line1 = lines!(ax1, sqrt.(res_udcg), color = :green,
               linewidth = 3,
               label = latexstring("DCG \$\$"))

line1 = lines!(ax1, log2.data[:resnorm], color = :black,
               linestyle = :dash,
               linewidth = 3,
               label = latexstring("CG with \$f_{d}\$"))

axislegend(ax1, position = :cb)
save(string("residual_",chopsuffix(fname, ".jld2"),".pdf"),fig1)


fig2 = Figure()
Axis(fig2[1, 1],
     xlabel = latexstring("\$x\$-axis"),
     ylabel = latexstring("\$y\$-axis"),
     aspect = DataAspect(),
     limits = (0.0,pi,0,pi))
cplot = CairoMakie.contourf!(fig2[1,1], x,y,ucg,colormap = :plasma)
Colorbar(fig2[1, 2], cplot)
save(string("solution_",chopsuffix(fname, ".jld2"),".pdf"),fig2)

