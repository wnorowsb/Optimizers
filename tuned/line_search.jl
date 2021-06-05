import Base.MathConstants:φ
function golden_section_search(f::Function, a::Float64, b::Float64;n=50)
    ρ = φ - 1
    d = ρ * b + (1 - ρ) * a
    yd = f(d)
    for i = 1:n - 1
        c = ρ * a + (1 - ρ) * b
        yc = f(c)
        if yc < yd
            b, d, yd = d, c, yc
        else
            a, b = b, c
        end
    end
    return a < b ? (a, b) : (b, a)
end

function bracket_minimum(f::Function; x::Float64=0.0, s::Float64=1e-2, k::Float64=2.0) 
    a::Float64, ya::Float64 = x, f(x)
    b::Float64, yb::Float64 = a + s, f(a + s)
    if yb > ya
        a, b = b, a
        ya, yb = yb, ya
    s = -s
    end
    while true
        c::Float64, yc::Float64 = b + s, f(b + s)
        if yc > yb
            return a < c ? (a, c) : (c, a)
        end
        a, ya, b, yb = b, yb, c, yc
        s *= k
    end
end

function line_search(φ::Function, φ′::Function, d)
    a, b = bracket_minimum(φ)
    # print(a)
    # print(b)
    # println(φ)
    x, y = golden_section_search(φ, a, b)
    x / 2 + y / 2
end