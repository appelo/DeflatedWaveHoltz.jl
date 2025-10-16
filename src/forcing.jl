
function set_gauss_forcing!(P2D::Prob2D,x0::Float64,y0::Float64)
    Nx = P2D.Nx
    Ny = P2D.Ny
    force  = P2D.force
    x_grid = P2D.x_grid
    y_grid = P2D.y_grid
    omega = P2D.omega
    for j = 1:Ny
        for i = 1:Nx
            force[i+(j-1)*Nx] =
                omega^2*exp(-omega^2*(
                    (x_grid[i]-x0)^2+(y_grid[j]-y0)^2))
        end
    end
end
