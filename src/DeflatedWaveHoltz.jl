module DeflatedWaveHoltz

using LinearAlgebra, SparseArrays, SummationByPartsOperators   

export DirichletProb2D 

export set_gauss_forcing!

export compute_a0, WHI_operator!

Prob2D = Union{DirichletProb2D}

include("dirichletproblem2d.jl")
include("forcing.jl")
include("waveholtzoperators.jl")

Base.show(io::IO, dp::DirichletProb2D) =
    print(io,"Dirichlet Problem at omega = ",
          dp.omega,",\n","Posed on [",
          dp.x_grid[1]-dp.hx,",",dp.x_grid[end]+dp.hx,"]x[",
          dp.y_grid[1]-dp.hy,",",dp.y_grid[end]+dp.hy,"]",
          ",\nUsing Nx = ",dp.Nx,", Ny = ",dp.Ny," gridpoints",
          ",\nThe number of timesteps per wave solve is Nt = ",dp.Nt,
          ",\nThe order of the method is ",dp.order,
          ",\nThe number of DOF are ",dp.N,".")

end
