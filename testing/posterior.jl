using Distributions, Plots, CustomPlots
using LinearAlgebra, SpecialFunctions
include("../functions.jl")


# ============================================================
# ======================   Very simple   =====================
# ============================================================

n = 10_000

# True
μ = 1
σ = 1/sqrt(16000)
Y = rand(Normal(μ, σ), n)
X = ones(n)

μ₀ = 0 # Prior mean
λ  = 0.1 # Prior precision
α  = 0 # Prior a
u  = 0 # Prior rate

x̄ = mean(Y)
S = mean((Y .- x̄).^2)
pΛ = Gamma(α + n/2, inv(u+0.5*(n*S+(λ*n*(x̄-μ₀)^2)/(λ+n))))
mode(pΛ)

Y'*Y + μ₀'*λ*μ₀ - (X'*Y+λ*μ₀)'*inv(X'*X+λ)'*(X'*Y+λ*μ₀)
n*S+(λ*n*(x̄-μ₀)^2)/(λ+n)
sum((Y .-x̄).^2) + n*λ/(n+λ)*(x̄-μ₀)

(λ*n*(x̄-μ₀))/(λ+n)
n*λ*(x̄-μ₀)/(n+λ)

histogram(rand(pΛ, 10_000))
vline!([σ^-2, 1/var(Y)])

(n*S+(λ*n*(x̄-μ₀)^2)/(λ+n))

τ = σ^-2
β = 2
(Y - X*β)'*(Y - X*β) + λ*(β - μ₀)^2

Y'*Y-2*Y'*X*β + β'*X'*X*β + β'*λ*β - 2λ*β'*μ₀ + μ₀'*λ*μ₀

(β - inv(X'*X+λ)*(Y'*X+λ*μ₀))'*(X'*X+λ)*(β - inv(X'*X+λ)*(Y'*X+λ*μ₀)) + Y'*Y + μ₀'*λ*μ₀ - (X'*Y+λ*μ₀)*inv(X'*X+λ)*(Y'*X+λ*μ₀)

Gamma(α + n/2, inv(u + 0.5(Y'*Y + μ₀'*λ*μ₀ - (X'*Y+λ*μ₀)*inv(X'*X+λ)*(Y'*X+λ*μ₀)))) |> mode


# ============================================================
# ======================   More Params   =====================
# ============================================================


d = 2
N = 10_000
βt = collect(range(-0.5,0.5,length=d))
σt = 1/sqrt(16000)

X = hcat(range(0.0, 10.0, length = N), range(-5, 5, length = N))
Y = X*βt + randn(N)*σt

β₀ = ones(d)*0.5
λ = I(d)*1
a = 1
u = 1

pΛ = Gamma(a+N/2, inv(u+0.5*(Y'*Y + β₀'*λ*β₀ - (X'*Y+λ*β₀)'*inv(X'*X+λ)*(X'*Y+λ*β₀))))
mode(pΛ)

sqrt(1/mean(pΛ))
sqrt(1/median(pΛ))
sqrt(1/mode(pΛ))

pβ = MvNormal(inv(X'*X+λ)*(X'*Y+λ*β₀), inv(X'*X+λ))

β = inv(X'*X+λ)*(X'*Y+λ*β₀)

1/var(Y - X*β)

β = rand(pβ)
β = inv(X'*X)*X'*Y

n = 5000
histogram(rand(pΛ, n), label = "")
vline!([(1/σt)^2], label = "True")
vline!([N/((Y - X*β)'*(Y - X*β))], label = "MLE")

histogram(rand(pΛ, n).^(-0.5))
vline!([σt, sqrt((Y - X*β)'*(Y - X*β)/N)])

inv(u+0.5*(Y'*Y - (X'*Y+λ*β₀)'*inv(X'*X+λ)*(X'*Y+λ*β₀))) |> clipboard

f = x -> x^(a+N/2+(d-2)/2)*exp(-x*(u+0.5*(Y'*Y - (X'*Y+λ*β₀)'*inv(X'*X+λ)*(X'*Y+λ*β₀))))
Z = int(f, 0, Inf)

fplot((0,0.3), x -> f(x)/Z)
histogram!(rand(pΛ, n), normalize=:pdf)
vline!([(1/σt)^2, N/((Y - X*β)'*(Y - X*β))])


n = 10_000
p = 10
d = 5
X = hcat(ones(n), randn(n, p-1))
βt = randn(p, d)
Y = X*βt + randn(size(Y))*1e-2

Y[1,:]

β' * X[1,:]

(Y - X*β) * inv(Σ)

X[1,:]'*βt

β = inv(X'*X)*X'*Y

[1, zeros(p-1)...]' * βt