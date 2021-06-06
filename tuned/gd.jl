mutable struct GD
    lr::Float64 
    GD() = new()
end

function init!(M::GD, lr)
    M.lr = lr
    return M
end

function step!(M::GD, f::Function, ∇f::Function, θ::Vector{Float64})
    θ′ = θ - ∇f(θ)::Vector{Float64} * M.lr
    return θ′
end