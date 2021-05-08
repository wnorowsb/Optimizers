include("bfgs.jl")
include("l-bfgs.jl")
include("gd.jl")

rosenbrock(x; a=1, b=5) = (a-x[1])^2 + b*(x[2] - x[1]^2)^2
rosenbrockGrad(x; a=1, b=5) = [2*(x[1]-a)-2*b*x[1]*(x[2]-x[1]^2), 2*b*(x[2] - x[1]^2)]

M=BFGS()
myBFGS = init!(M,[0.75, 2.0])
θ = [0.75, 2.0]
println("BFGS")
for i in 1:5
    global θ
    println(θ)
    println(rosenbrock(θ))
    θ′ = step!(myBFGS,rosenbrock,rosenbrockGrad, θ)
    θ = θ′
end

M=LBFGS()
myLBFGS = init!(M,2) #im wikeszy 2 parametr tym wiecej pamieci i lepsze wyniki
θ2 = [0.75, 2.0]
println("L-BFGS")
for i in 1:5
    global θ2
    println(θ2)
    println(rosenbrock(θ2))
    θ′2 = step!(myLBFGS,rosenbrock,rosenbrockGrad, θ2)
    θ2 = θ′2
end

M=GD()
myGD = init!(M,0.01) #im wikeszy 2 parametr tym wiecej pamieci i lepsze wyniki
θ3 = [0.75, 2.0]
println("GRADIENT DESCENT")
for i in 1:5
    global θ3
    println(θ3)
    println(rosenbrock(θ3))
    θ′3 = step!(myGD,rosenbrock,rosenbrockGrad, θ3)
    θ3 = θ′3
end