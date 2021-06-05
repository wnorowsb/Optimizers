include("line_search.jl")
using LinearAlgebra

mutable struct BFGS
    Q::Matrix{Float64} # approximation of
    # Hessian matrix inverse
    BFGS() = new()
end

function init!(M::BFGS, θ)
    p = length(θ)
    # initiated with an identity matrix
    M.Q = Matrix(1.0I, p, p)
    return M
end

function step!(M::BFGS, f::Function, ∇f::Function, θ::Vector{Float64})
    Q = M.Q
    g = ∇f(θ)
    d::Vector{Float64} = -Q * g
    φ = α -> f(θ + α * d)
    φ′ = α -> ∇f(θ + α * d) ⋅ d
    α = line_search(φ, φ′, d)
    # println(α)
    # println(d)
    θ′ = θ + α * d
    # println(θ)
    # println(θ′)
    g′ =  ∇f(θ′)
    δ = θ′ - θ
    γ = g′ - g
    Q[:] = Q - (δ * γ' * Q + Q * γ * δ') / (δ' * γ) +
              (1.0 + (γ' * Q * γ) / (δ' * γ)) *
                (δ * δ') / (δ' * γ)
    return θ′
end
