mutable struct GD_basic
    lr # approximation of
    # Hessian matrix inverse
    GD_basic() = new()
end

function init!(M::GD_basic, lr)
    M.lr = lr
    return M
end

function step!(M::GD_basic, f, ∇f, θ)
    θ′ = θ - ∇f(θ) * M.lr
    return θ′
end