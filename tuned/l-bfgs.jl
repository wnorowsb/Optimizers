include("line_search.jl")
using LinearAlgebra
mutable struct LBFGS
    m::Int
    δs::Vector{Vector{Float64}}
    γs::Vector{Vector{Float64}}
    qs::Vector{Vector{Float64}}
    LBFGS() = new()
end

function init!(M::LBFGS, m)
    M.m = m
    M.δs = []
    M.γs = []
    M.qs = []
    return M
end

function step!(M::LBFGS, f::Function, ∇f::Function, θ::Vector{Float64})
    δs, γs, qs = M.δs, M.γs, M.qs
    m, g::Vector{Float64} = length(δs), ∇f(θ)
    d::Vector{Float64} = -g
    if m > 0
        q::Vector{Float64} = g
        for i in m:-1:1
            qs[i] = copy(q)
            q -= (δs[i] ⋅ q) / (γs[i] ⋅ δs[i]) * γs[i]
        end
        z = (γs[m] .* δs[m] .* q) / (γs[m] ⋅ γs[m])
        for i in 1:+1:m
            z += δs[i] * (δs[i] ⋅ qs[i] - γs[i] ⋅ z) / (γs[i] ⋅ δs[i])
        end
        d = -z;
    end
    φ = α -> f(θ + α * d)::Float64; φ′ = α -> ∇f(θ + α * d)::Vector{Float64} ⋅ d
    α = line_search(φ, φ′, d)
    θ′ = θ + α * d; g′::Vector{Float64} = ∇f(θ′)
    δ = θ′ - θ; γ = g′ - g
    push!(δs, δ); push!(γs, γ); push!(qs, zero(θ))
    while length(δs) > M.m
        popfirst!(δs); popfirst!(γs); popfirst!(qs)
    end
    return θ′
end