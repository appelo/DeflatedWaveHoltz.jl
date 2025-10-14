using DeflatedWaveHoltz
using LinearAlgebra, SparseArrays, SummationByPartsOperators, IterativeSolvers, AlgebraicMultigrid, LinearMaps, ArnoldiMethod, Plots


function t()
    order = 4
    ep_tol = 1e-3

    omega = 15.3

    xmin = -1.0
    xmax =  1.0
    ymin = -2.0
    ymax =  2.0

    DP  = DeflatedWaveHoltz.DirichletProb2D(omega,xmin,xmax,ymin,ymax,order,ep_tol)

    set_gauss_forcing!(DP,0.1,0.2)
    contour(DP.x_grid,DP.y_grid,transpose(reshape(DP.force,DP.Nx,DP.Ny)),
            aspect_ratio = 1.0)

    show(DP)
    uin = zeros(DP.N)
    uproj = zeros(DP.N)

    A_whi_ex!(y,x) = S_WHI_operator_hom!(y,x,DP)
    ALM_WHI = LinearMap(A_whi_ex!,DP.N,DP.N,issymmetric = true,ismutating=true,isposdef=true)

    b = zeros(DP.N)
    ucg = zeros(DP.N)
    WHI_operator!(b,uin,DP)
    ucg, log1 = cg(ALM_WHI,b,log=true,reltol=1e-12,
                  verbose=true,
                   maxiter=DP.N)
    pl = contour(DP.x_grid,DP.y_grid,transpose(reshape(ucg,DP.Nx,DP.Ny)),
                 aspect_ratio = 1.0)
    display(pl)
    println(log)
    return pl
    #=    
    for it = 1:100
        WHI_operator!(uproj,uin,DP)
        pl = contour(DP.x_grid,DP.y_grid,transpose(reshape(uproj,DP.Nx,DP.Ny)),
                     aspect_ratio = 1.0)
        display(pl)
        uin .= uproj
    end
=#
end

