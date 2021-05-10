include("bfgs.jl")
include("l-bfgs.jl")
include("gd.jl")
include("functions.jl")

rosenbrock(x; a=1, b=5) = (a-x[1])^2 + b*(x[2] - x[1]^2)^2
rosenbrockGrad(x; a=1, b=5) = [2*(x[1]-a)-2*b*x[1]*(x[2]-x[1]^2), 2*b*(x[2] - x[1]^2)]

function testBFGS(iter)
    M=BFGS()
    myBFGS = init!(M,[0.75, 2.0])
    θ = [0.75, 2.0]
    println(θ)
    println(rosenbrock(θ))
    for i in 1:iter
        θ′ = step!(myBFGS, rosenbrock, rosenbrockGrad, θ)
        if rosenbrock(θ) - rosenbrock(θ′) < 0.0001
            println("Liczba iteracji: " * string(i))
            break
        end
        θ = θ′
        println(θ)
        println(rosenbrock(θ))
    end
end
    
function testLBFGS(iter)
    M=LBFGS()
    myLBFGS = init!(M,2) #im wikeszy 2 parametr tym wiecej pamieci i lepsze wyniki
    θ2 = [0.75, 2.0]
    println(θ2)
    println(rosenbrock(θ2))
    for i in 1:iter
        θ′2 = step!(myLBFGS,rosenbrock,rosenbrockGrad, θ2)
        if rosenbrock(θ2) - rosenbrock(θ′2) < 0.0001
            println("Liczba iteracji: " * string(i))
            break
        end
        θ2 = θ′2
        println(θ2)
        println(rosenbrock(θ2))
    end
end

function testGD(iter)
    M=GD()
    myGD = init!(M,0.019) 
    θ3 = [0.75, 2.0]
    println(θ3)
    println(rosenbrock(θ3))
    for i in 1:iter
        θ′3 = step!(myGD,rosenbrock,rosenbrockGrad, θ3)
        if rosenbrock(θ3) - rosenbrock(θ′3) < 0.0001
            println("Liczba iteracji: " * string(i))
            break
        end
        θ3 = θ′3
        println(θ3)
        println(rosenbrock(θ3))
    end
end

@time testLBFGS(15)
@time testBFGS(15)
@time testGD(15)