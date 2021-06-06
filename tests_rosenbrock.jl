include("tuned/bfgs.jl")
include("tuned/l-bfgs.jl")
include("tuned/gd.jl")
# include("basic/bfgs.jl")
# include("basic/l-bfgs.jl")
# include("basic/gd.jl")
using BenchmarkTools
using ForwardDiff
# include("functions.jl")

rosenbrock(x; a=1, b=5) = (a - x[1])^2 + b * (x[2] - x[1]^2)^2
rosenbrockGrad(x; a=1, b=5) = [2 * (x[1] - a) - 2 * b * x[1] * (x[2] - x[1]^2), 2 * b * (x[2] - x[1]^2)]

function ackley(x, a=20, b=0.2, c=2π)
    d = length(x)
    return -a * exp(-b * sqrt(sum(x.^2) / d)) -
    exp(sum(cos.(c * xi) for xi in x) / d) + a + exp(1)
end

ackleyGrad = x -> ForwardDiff.gradient(ackley, x);

function testBFGS(iter)
    M = BFGS()
    myBFGS = init!(M, [0.75, 2.0]) # im wikeszy 2 parametr tym wiecej pamieci i lepsze wyniki
    θ = [0.75, 2.0]
    # println("BFGS")
    # println(θ)
    # println(rosenbrock(θ))
    for i in 1:iter
        θ′ = step!(myBFGS, rosenbrock, rosenbrockGrad, θ)
        if rosenbrock(θ) - rosenbrock(θ′) < 0.000005
            # println("Liczba iteracji: " * string(i))
            break
        end
        θ = θ′
        # println(θ)
        # println(rosenbrock(θ))
    end
end
    
function testLBFGS(iter)
    M = LBFGS()
    myLBFGS = init!(M, 2) # im wikeszy 2 parametr tym wiecej pamieci i lepsze wyniki
    θ2 = [0.75, 2.0]
    # println("LBFGS")
    # println(θ2)
    # println(rosenbrock(θ2))
    for i in 1:iter
        θ′2 = step!(myLBFGS, rosenbrock, rosenbrockGrad, θ2)
        if rosenbrock(θ2) - rosenbrock(θ′2) < 0.000005
            # println("Liczba iteracji: " * string(i))
            break
        end
        θ2 = θ′2
        # println(θ2)
        # println(rosenbrock(θ2))
    end
end

function testGD(iter)
    M = GD()
    myGD = init!(M, 0.019) 
    θ3 = [0.75, 2.0]
    # println("GD")
    # println(θ3)
    # println(rosenbrock(θ3))
    for i in 1:iter
        θ′3 = step!(myGD, rosenbrock, rosenbrockGrad, θ3)
        if rosenbrock(θ3) - rosenbrock(θ′3) < 0.000005
            # println("Liczba iteracji: " * string(i))
            break
        end
        θ3 = θ′3
        # println(θ3)
        # println(rosenbrock(θ3))
    end

end

# testLBFGS(2)
@btime testBFGS(20)
@btime testLBFGS(20)
@btime testGD(20)
println("Koniec testów")