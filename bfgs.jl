using LinearAlgebra
import Base.MathConstants: φ
function golden_section_search(f,a,b;n=50)
    ρ = φ - 1
    d = ρ * b + (1 - ρ)*a
    yd = f(d)
    for i = 1:n-1
        c = ρ*a + (1 - ρ)*b
        yc = f(c)
        if yc < yd
            b, d, yd = d, c, yc
        else
            a, b = b, c
        end
    end
    return a < b ? (a, b) : (b, a)
end

function bracket_minimum(f; x=0, s=1e-2, k=2.0)
    a, ya = x, f(x)
    b, yb = a + s, f(a + s)
    if yb > ya
        a, b = b, a
        ya, yb = yb, ya
        s = -s
    end
    while true
        c, yc = b + s, f(b + s)
        if yc > yb
            return a < c ? (a, c) : (c, a)
        end
        a, ya, b, yb = b, yb, c, yc
        s *= k
    end
end

function line_search(φ, φ′, d)
    a, b = bracket_minimum(φ)
    x, y = golden_section_search(φ, a, b)
    x/2 + y/2
end

mutable struct BFGS
    Q # approximation of
    # Hessian matrix inverse
    BFGS() = new()
end

function init!(M::BFGS, θ)
    p = length(θ)
    # initiated with an identity matrix
    M.Q = Matrix(1.0I, p, p)
    return M
end

function step!(M::BFGS, f, ∇f, θ)
    Q = M.Q
    g = ∇f(θ)
    d=-Q*g
    φ = α -> f(θ + α*d)
    φ′ = α -> ∇f(θ + α*d)⋅d
    α = line_search(φ, φ′, d)
    # println(α)
    # println(d)
    θ′ = θ + α*d
    #println(θ)
    #println(θ′)
    g′ =  ∇f(θ′)
    δ = θ′ - θ
    γ = g′ - g
    Q[:]= Q - (δ*γ'*Q+Q*γ*δ')/(δ'*γ) +
              (1.0 + (γ'*Q*γ)/(δ'*γ)) *
                (δ*δ')/(δ'*γ)
    return θ′
end



rosenbrock(x; a=1, b=5) = (a-x[1])^2 + b*(x[2] - x[1]^2)^2
rosenbrockGrad(x; a=1, b=5) = [2*(x[1]-a)-2*b*x[1]*(x[2]-x[1]^2), 2*b*(x[2] - x[1]^2)]

M=BFGS()
myBFGS = init!(M,[0.75, 2.0])
θ = [0.75, 2.0]
for i in 1:5
    global θ
    println(θ)
    println(rosenbrock(θ))
    θ′ = step!(myBFGS,rosenbrock,rosenbrockGrad, θ)
    θ = θ′
end
