using Dates, StatsBase, DataFrames
import StatsBase: sample
import Base: range, String

Base.String(s::Minute) = "$(s.value)T"

function Base.range(r::Vector{DateTime}; step::TimePeriod=Minute(1))
    return range(r[1], r[end], step = step)
end

function StatsBase.sample(a::DateTime, b::DateTime; step::TimePeriod=Minte(1))
    return range(a, b, step = step) |> rand
end

function findmissing(r::Vector{DateTime}; step::TimePeriod=Minute(1))
    return findall(t -> t ∉ r, range(r, step=step))
end

function filtermissing(r::Vector{DateTime}; step::TimePeriod=Minute(1))
    return filter(t -> t ∉ r, range(r, step = step))
end

function fillMissing!(data::DataFrame; step::TimePeriod = Minute(1))
    for (i, idx) in enumerate(findmissing(data.t, step = step))
        a = Dict(k => v for (k,v) in pairs(data[idx - i,:]))
        a[:t] = data.t[1] + (idx-1)*step
        push!(data, a)
    end
    sort!(data, :t)
end

function designMatrix(X::AbstractArray)
    return hcat(repeat([1], size(X,1)), X)
end

function int(f::Function, a::Number, b::Number; h::Number = 25)
    if !isinf(a) && !isinf(b)
        g = x -> a + 0.5(tanh(π/2 * sinh(x/h))+1) * (b-a)
        dg = x -> (b-a)/4*cosh(x/h)*(sech(π/2*sinh(x/h)))^2
    elseif isinf(a) && isinf(b)
        g = x -> π * sinh(x/h)
        dg = x -> cosh(x/h)
    elseif isinf(a)
        g = x -> log(0.5*tanh(π/2*sinh(x/h))+0.5)+b
        dg = x -> cosh(x/h)/(1+exp(π*sinh(x/h)))
    elseif isinf(b)
        g = x -> a-log(0.5*tanh(π/2*sinh(x/h))+0.5)
        dg = x -> cosh(x/h)/(1+exp(π*sinh(x/h)))
    end
    F = x -> (f∘g)(x)
    return (dg(0)*F(0) + sum(i -> dg(i)*F(i)+dg(-i)*F(-i), 1:ceil(Int, 3.1h)))*π/h
end

function ±(x, y)
    return x .+ [1, -1] .* y
end
function ∓(x, y)
    return x .+ [-1, 1] .* y
end

function (twapBias)(step=Minute(1))
    return function twapBias(t,as,bs)
        b = @. (as-bs)/(as+bs)
        tw = diff([round(t[1], step, RoundDown) ; t])
        tw = [x.value for x in tw]
        return sum(tw) == 0 ? mean(b) : tw'*b/sum(tw)
    end
end
function (twapPrice)(step=Minute(1))
    return function f(t,as,ap,bs,bp)
        b = @. (as*ap+bs*bp)/(as+bs)
        tw = diff([round(t[1], step, RoundDown) ; t])
        tw = [x.value for x in tw]
        return sum(tw) == 0 ? mean(b) : tw'*b/sum(tw)
    end
end
function (twa)(step=Minute(1))
    return function f(t,b)
        tw = diff([round(t[1], step, RoundDown) ; t])
        tw = [x.value for x in tw]
        return sum(tw) == 0 ? mean(b) : tw'*b/sum(tw)
    end
end
function (Quantile)(q)
    return x -> quantile(x, q)
end

function prog(tm::DateTime, p::Period)
    L = round(tm, p, RoundDown)
    U = round(tm, p, RoundUp)
    L != U ? (tm-L)/(U-L) : 0
end