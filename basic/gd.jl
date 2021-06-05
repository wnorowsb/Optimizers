mutable struct GD
    lr # approximation of
    # Hessian matrix inverse
    GD() = new()
end

function init!(M::GD, lr)
    M.lr = lr
    return M
end

function step!(M::GD, f, ∇f, θ)
    θ′ = θ - ∇f(θ) * M.lr
    return θ′
end