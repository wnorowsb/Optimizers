include("tuned/bfgs.jl")
include("tuned/l-bfgs.jl")
include("tuned/gd.jl")
using BenchmarkTools
using ForwardDiff

function himmelblau(x)
     return(x[1]^2+x[2]-11)^2+(x[1]+x[2]^2-7)^2
end

himmelblauGrad = x -> ForwardDiff.gradient(himmelblau, x);

function testBFGS(iter)
    M = BFGS()
    myBFGS = init!(M, [0.75, 2.0]) # im wikeszy 2 parametr tym wiecej pamieci i lepsze wyniki
    θ = [0.75, 2.0]
    # println("BFGS")
    # println(θ)
    # println(himmelblau(θ))
    for i in 1:iter
        θ′ = step!(myBFGS, himmelblau, himmelblauGrad, θ)
        if himmelblau(θ) - himmelblau(θ′) < 0.000005
            # println("Liczba iteracji: " * string(i))
            break
        end
        θ = θ′
        # println(θ)
        # println(himmelblau(θ))
    end
end
    
function testLBFGS(iter)
    M = LBFGS()
    myLBFGS = init!(M, 2) # im wikeszy 2 parametr tym wiecej pamieci i lepsze wyniki
    θ2 = [0.75, 2.0]
    # println("LBFGS")
    # println(θ2)
    # println(himmelblau(θ2))
    for i in 1:iter
        θ′2 = step!(myLBFGS, himmelblau, himmelblauGrad, θ2)
        if himmelblau(θ2) - himmelblau(θ′2) < 0.000005
            # println("Liczba iteracji: " * string(i))
            break
        end
        θ2 = θ′2
        # println(θ2)
        # println(himmelblau(θ2))
    end
end

function testGD(iter)
    M = GD()
    myGD = init!(M, 0.019) 
    θ3 = [0.75, 2.0]
    # println("GD")
    # println(θ3)
    # println(himmelblau(θ3))
    for i in 1:iter
        θ′3 = step!(myGD, himmelblau, himmelblauGrad, θ3)
        if himmelblau(θ3) - himmelblau(θ′3) < 0.000005
            # println("Liczba iteracji: " * string(i))
            break
        end
        θ3 = θ′3
        # println(θ3)
        # println(himmelblau(θ3))
    end

end

@btime testBFGS(20)
@btime testLBFGS(20)
@btime testGD(20)
println("Koniec testów")