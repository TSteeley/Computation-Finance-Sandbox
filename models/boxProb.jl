using Distributions, Plots, CustomPlots, SpecialFunctions
include("../int.jl")

σ = 1

n = 2
N = 1_250_000

8*N*n < 1e9 ? println("$(round(8*N*n/1e6, digits = 3)) Mb") : error("Too Large")

U, L = 1, -1

X = cumsum(randn(N, n) .* σ/√n, dims = 2)

for x in eachrow(X)
    idx = findfirst((x .< L) .| (x .> U))
    if ! isnothing(idx) && (x[idx] < L)
        x[idx:end] .= L
    elseif ! isnothing(idx) && (x[idx] > U)
        x[idx:end] .= U
    end
end


#############################
####  Plot Distribution  ####
#############################

x = @view X[:,end]

histogram(x, normalize = :pdf, bins = 100)

mean(x .== U)      # Hit upper limit
mean(x .== L)      # Hit lower limit
mean(L .< x .< U)  # Never hit either

histogram(x[L .< x .< U], normalize = :pdf, bins = 20)

mean(x)
mean(x[L .< x .< U])


#######################
####  Plot Traces  ####
#######################

p = plot()
[plot!(p, x, label = "", la = 0.2, lc = :black) for x in eachrow(X)]
hline!(p, [L, U], label = "", lc = :red)
p



# ============================================================
# ===========   Messing around with convolutions   ===========
# ============================================================

V = 1/3
f(x) = exp(-x^2/(2V))/sqrt(2π*V)

L, U = -2, 1

fplot((L, U), t -> int(x -> f(x)*f(t-x), L, U), lw = 3)
histogram!(x[L .< x .< U], normalize = :pdf, bins = 200)

fplot((-5, 5), t -> int(y -> int(x -> f(x)*f(y-x), L, U)*exp(-im*t*y), -Inf, Inf) |> real)

fplot!((-5, 5), t -> int(x -> f(x)*int(y -> f(y-x)*exp(-im*t*y), -Inf, Inf), L, U) |> real)

fplot!((-5, 5), y -> int(x -> f(x)*exp(-im*x*y), L, U)*int(u -> f(u)*exp(-im*u*y), -Inf, Inf)|> real)

fplot((-5, 5), y -> int(u -> f(u)*exp(-im*u*y), -Inf, Inf)|> real)
fplot!((-5, 5), y -> exp(-V*y^2/2))

fplot((-5, 5), y -> int(x -> f(x)*exp(-im*x*y), L, U)|> real)
fplot!((-5, 5), y -> exp(-V*y^2/2)*(erf((U+im*V*y)/sqrt(2V))-erf((L+im*V*y)/sqrt(2V)))/2 |> real)

fplot!((-5, 5), y -> exp(-V*y^2)*(erf((U+im*V*y)/sqrt(2V))-erf((L+im*V*y)/sqrt(2V)))/2 |> real)

int(x -> int(y -> exp(im*y*x)*exp(-V*y^2/2)^3*((erf((U+im*V*y)/sqrt(2V))-erf((L+im*V*y)/sqrt(2V)))/2), -Inf, Inf) / 2π |> real, L, U)

t = 1
int(z -> int(y -> int(x -> f(x)*f(y-x)*f(z-y)*exp(-im*t*z), L, U), L, U), L, U)

int(x -> f(x)*exp(-im*t*x), L, U)*int(x -> f(x)*exp(-im*t*x), -Inf, Inf)

int(y -> int(x -> f(x)*f(y-x)*exp(-im*t*y), L, U), -Inf, Inf)