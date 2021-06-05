include("line_search_basic.jl")
using LinearAlgebra

mutable struct BFGS_basic
    Q # approximation of
    # Hessian matrix inverse
    BFGS_basic() = new()
end

function init!(M::BFGS_basic, θ)
    p = length(θ)
    # initiated with an identity matrix
    M.Q = Matrix(1.0I, p, p)
    return M
end

function step!(M::BFGS_basic, f, ∇f, θ)
    Q = M.Q
    g = ∇f(θ)
    d = -Q * g
    φ = α -> f(θ + α * d)
    φ′ = α -> ∇f(θ + α * d) ⋅ d
    α = line_search_basic(φ, φ′, d)
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
