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