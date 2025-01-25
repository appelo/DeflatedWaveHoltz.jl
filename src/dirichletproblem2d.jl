mutable struct DirichletProb
    omega::Float64

    function DirichletProb(
        omega::Float64)
        new(omega)
    end
end
