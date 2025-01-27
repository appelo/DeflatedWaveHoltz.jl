using DeflatedWaveHoltz
using Plots

function t()
    order = 4
    ep_tol = 1e-3

    omega = 15.3

    xmin = -1.0
    xmax =  1.0
    ymin = -2.0
    ymax =  2.0

    DP  = DeflatedWaveHoltz.DirichletProb2D(omega,xmin,xmax,ymin,ymax,order,ep_tol)

    set_gauss_forcing!(DP,0.0,0.0)
    contour(DP.x_grid,DP.y_grid,transpose(reshape(DP.force,DP.Nx,DP.Ny)),
            aspect_ratio = 1.0)

    show(DP)
    uin = zeros(DP.N)
    uproj = zeros(DP.N)

    for it = 1:100
        WHI_operator!(uproj,uin,DP)
        pl = contour(DP.x_grid,DP.y_grid,transpose(reshape(uproj,DP.Nx,DP.Ny)),
                     aspect_ratio = 1.0)
        display(pl)
        uin .= uproj
    end
end

t()
