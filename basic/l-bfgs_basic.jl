include("line_search_basic.jl")
using LinearAlgebra
mutable struct LBFGS_basic
    m
    δs
    γs
    qs
    LBFGS_basic() = new()
end

function init!(M::LBFGS_basic, m)
    M.m = m
    M.δs = []
    M.γs = []
    M.qs = []
    return M
end

function step!(M::LBFGS_basic, f, ∇f, θ)
    δs, γs, qs = M.δs, M.γs, M.qs
    m, g = length(δs), ∇f(θ)
    d = -g
    if m > 0
        q = g
        for i in m:-1:1
            qs[i] = copy(q)
            q -= (δs[i]⋅q) / (γs[i]⋅δs[i]) * γs[i]
        end
        z = (γs[m] .* δs[m] .* q) / (γs[m]⋅γs[m])
        for i in 1:+1:m
            z += δs[i]*(δs[i]⋅qs[i]-γs[i]⋅z)/(γs[i]⋅δs[i])
        end
        d = -z;
    end
    φ = α -> f(θ + α*d); φ′ = α -> ∇f(θ + α*d)⋅d
    α = line_search_basic(φ, φ′, d)
    θ′ = θ + α*d; g′ = ∇f(θ′)
    δ = θ′ - θ; γ = g′ - g
    push!(δs, δ); push!(γs, γ); push!(qs, zero(θ))
    while length(δs) > M.m
        popfirst!(δs); popfirst!(γs); popfirst!(qs)
    end
    return θ′
end