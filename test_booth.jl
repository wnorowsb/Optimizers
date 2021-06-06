include("tuned/bfgs.jl")
include("tuned/l-bfgs.jl")
include("tuned/gd.jl")
using BenchmarkTools
using ForwardDiff

function booth(x)
     return (x[1]+2*x[2]-7)^2+(2*x[1]+x[2]-5)^2
end

boothGrad = x -> ForwardDiff.gradient(booth, x);

function testBFGS(iter)
    M = BFGS()
    myBFGS = init!(M, [0.75, 2.0]) # im wikeszy 2 parametr tym wiecej pamieci i lepsze wyniki
    θ = [0.75, 2.0]
    # println("BFGS")
    # println(θ)
    # println(booth(θ))
    for i in 1:iter
        θ′ = step!(myBFGS, booth, boothGrad, θ)
        if booth(θ) - booth(θ′) < 0.000005
            # println("Liczba iteracji: " * string(i))
            break
        end
        θ = θ′
        # println(θ)
        # println(booth(θ))
    end
end
    
function testLBFGS(iter)
    M = LBFGS()
    myLBFGS = init!(M, 2) # im wikeszy 2 parametr tym wiecej pamieci i lepsze wyniki
    θ2 = [0.75, 2.0]
    # println("LBFGS")
    # println(θ2)
    # println(booth(θ2))
    for i in 1:iter
        θ′2 = step!(myLBFGS, booth, boothGrad, θ2)
        if booth(θ2) - booth(θ′2) < 0.000005
            # println("Liczba iteracji: " * string(i))
            break
        end
        θ2 = θ′2
        # println(θ2)
        # println(booth(θ2))
    end
end

function testGD(iter)
    M = GD()
    myGD = init!(M, 0.019) 
    θ3 = [0.75, 2.0]
    # println("GD")
    # println(θ3)
    # println(booth(θ3))
    for i in 1:iter
        θ′3 = step!(myGD, booth, boothGrad, θ3)
        if booth(θ3) - booth(θ′3) < 0.000005
            # println("Liczba iteracji: " * string(i))
            break
        end
        θ3 = θ′3
        # println(θ3)
        # println(booth(θ3))
    end

end

@btime testBFGS(20)
@btime testLBFGS(20)
@btime testGD(20)
println("Koniec testów")