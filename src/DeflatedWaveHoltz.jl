module DeflatedWaveHoltz

using LinearAlgebra, SparseArrays, SummationByPartsOperators, IterativeSolvers, AlgebraicMultigrid

include("dirichletproblem2d.jl")
export DirichletProb2D 
Prob2D = Union{DirichletProb2D}

include("dirichletproblem2di.jl")
export DirichletProb2Di 
Prob2D = Union{DirichletProb2D,DirichletProb2Di}


include("forcing.jl")
export set_gauss_forcing!

include("waveholtzoperators.jl")
export compute_a0, WHI_operator!, WHI_operator_hom!, WHI_operator_i!, WHI_operator_homi! 
export S_WHI_operator_hom!, S_WHI_operator_homi!
include("deflationfunctions.jl")
export epair_1d_laplace, bfunex

Base.broadcastable(x::Prob2D) =  Ref(x)

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
