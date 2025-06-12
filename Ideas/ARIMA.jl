using Distributions, StatsBase
using Plots, CustomPlots, ProgressBars
using FinData, Dates, DataFrames
using LinearAlgebra, SpecialFunctions
include("../functions.jl")

function Format_TS_data(TS::Vector{T}, P::Int, D::Int) where T <: Real
    if D != 0
        for _ in 1:D
            TS = diff(TS)
        end
    end

    X = hcat(
        repeat([1], length(TS)-P), 
        [TS[p:end-P+p-1] for p in 1:P]...
    )
    Y = TS[P+1:end]
    return X, Y
end

# ============================================================
# ==================   Load and Prep Data   ==================
# ============================================================

tStep = Minute(5)
data = loadCryptoQuotes("BTC")
filter!(d -> d.as .!= 0, data)

transform!(data, :t => ByRow(t -> round(t, tStep, RoundDown)) => :t)

minData = combine(groupby(data, :t),
    :bp => last => :bp,
    # :ap => mean => :ap,
)

trainPeriod = Week(3)
testPeriod = Week(1)

r = sample(minData.t[1], minData.t[end]-(trainPeriod+testPeriod+Week(1)), step = tStep)
train = filter(x -> r .≤ x.t .< r+trainPeriod, minData)
test = filter(x -> r+trainPeriod .≤ x.t .< r+trainPeriod+testPeriod, minData)

fillMissing!(train, step = tStep)
fillMissing!(test, step = tStep)

transform!(train, :bp => ByRow(log) => :X)
transform!(test, :bp => ByRow(log) => :X)

plot(train.t, train.X)
plot!(test.t, test.X)

# ============================================================
# ======================   ARIMA model   =====================
# ============================================================

P, D, Q = 5, 1, 1

N = length(train.X)-P-D
M = P+1

X, Y = Format_TS_data(train.X, P, D)
N, M = size(X)

β₀ = ones(M)*0
λ = I(M)
a = 1
u = 0

pΛ = Gamma(a + N/2, inv(u+0.5*(Y'*Y + β₀'*λ*β₀ - (X'*Y+λ*β₀)'*inv(X'*X+λ)*(X'*Y+λ*β₀))))
pβ = Λ -> MvNormal(inv(X'*X+λ)*(X'*Y+λ*β₀), inv(Λ*(X'*X+λ)) |> Hermitian)

mode(pΛ)
Λ = rand(pΛ)
β = rand(pβ(Λ))
N/((Y - X*β)'*(Y - X*β))


histogram(rand(pΛ, 10_000))
vline!([var(Y - X*β)^-1])

# Predict with model
y = zeros(length(test.t)-D)
# y[1:P] .= test.X[1:P]
y[1:P] .= diff(test.X[1:P+D])

for i in P+1:length(y)
    x = vcat(1, y[i-P:i-1])
    Λ = rand(pΛ)
    β = rand(pβ(Λ))
    y[i] = rand(Normal(x'*β,1/sqrt(Λ)))
end

plot(test.t, test.X, label = "True")
plot!(test.t, cumsum(vcat(test.X[1], y)), label = "Model")

# D = 0 
plot(test.t, test.X, label = "True")
plot!(test.t, y)

# D = 1
plot(test.t[1:end-D], diff(test.X), label = "True")
plot!(test.t[1:end-D], y, la = 0.2, label = "Model")

histogram(diff(test.X), normalize = :pdf)
histogram!(y, normalize = :pdf)

n = 5000
v = zeros(n)
for i in 1:n
    v[i] = rand(pβ(rand(pΛ))) |> x -> x[1] / sum(x[2:end])
end

histogram(v[-1 .< v .< 1])
vline!([mean(v)])

vline!([inv(X'*X)*X'*Y |> x -> x[1] / sum(x[2:end])])


# ============================================================
# ==============   Variable Precision model   ================
# ============================================================

# Model: yᵢ = xᵢ'*β + ε/sqrt(Λ(xᵢ))

X, Y = Format_TS_data(train.X, P, D)
N, M = size(X)

# We have two functions to work with
# yᵢ = xᵢᵀβ + εΛ(xᵢ)^(-1/2)
# -2ln(|yᵢ-xᵢᵀβ|) = xᵢᵀγ + ε
# Λ(xᵢ) = exp(xᵢᵀγ)

# yᵢ ∼ N( xᵢᵀβ, Λ(xᵢ)^(-1/2) )
# -2ln(|yᵢ-xᵢᵀβ|) ∼ 

# Likelihood

# β
β₀ = ones(M)*0
λ = I(M) * 50
pβ = Λ -> MvNormal(β₀, inv(Λ*λ) |> Hermitian)
cβ = Λ -> MvNormal(inv(X'*X+λ)*(X'*Y+λ*β₀), inv(Λ*(X'*X+λ)) |> Hermitian)


# γ
ℓ = (Z, γ, V) -> (a - 1) * sum(X*γ) - exp.(X*γ)
γ₀ = ones(M)*0
α = I(M)*1
u = 1e-6
a = 1

pγ = (V) -> MvNormal(γ₀, inv(V*α) |> Hermitian)
qγ = (Z,V) -> MvNormal(inv(X'*X+α)*(X'*Y+α*γ₀), inv(V*(X'*X+α)) |> Hermitian)

pV = Gamma(a, 1/u)
qV = Z -> Gamma(a + N/2, inv(u+0.5*(Z'*Z + γ₀'*α*γ₀ - (X'*Z+α*γ₀)'*inv(X'*X+α)*(X'*Z+α*γ₀))))



nSteps = 5_000
V = rand(pV)
γ = randn(M)
β = randn(M)
Λ = 1/var(Y - X*β)

Vlog = Vector{Vector{Float64}}(undef, 0)
γlog = Vector{Vector{Float64}}(undef, 0)
βlog = Vector{Vector{Float64}}(undef, 0)

for i in 1:nSteps |> ProgressBar
    βp = rand(cβ(Λ))
    push!(βlog, β)

    Z = -2*log.(abs.(Y - X*β))
    γp = rand(cγ(Z,V))
    if log(rand()) ≤ ℓ2(β,γp,a,Z)-ℓ2(β,γ,a,Z)-logpdf(qγ(β,γp),γ)+logpdf(qγ(β,γ),γp)
        γ = γp
        push!(γlog, γ)
    end
end

βplot = hcat(βlog...)
γplot = hcat(γlog...)

plot(βplot')
plot(γplot')

# Predict with model
y = zeros(length(test.t)-D)
# y[1:P] .= test.X[1:P]
y[1:P] .= diff(test.X[1:P+D])

βrange = range(round(Int, 0.2length(βlog)),length(βlog))
γrange = range(round(Int, 0.2length(γlog)),length(γlog))

iters = 1000
yMean = zeros(iters, length(y))
for j in 1:iters
    for i in P+1:length(y)
        x = vcat(1, y[i-P:i-1])
        μ = x'*βlog[rand(βrange)]
        σ = exp(x'*γlog[rand(γrange)]/2)
        y[i] = rand(Normal(μ,σ))
    end
    yMean[j,:] .= y
end


plot(test.t, test.X, label = "True")
plot!(test.t, cumsum(vcat(test.X[1], mean(yMean, dims = 1)[:])), label = "Model Mean", ribbon = 1.96 * std(yMean, dims = 1)[:])
plot!(test.t, cumsum(vcat(test.X[1], y)), label = "Model")



# Likelihood
L = gamma(a)^-N *exp(a*sum(Y2 - X*γ)-sum(exp.(Y2 - X*γ)))
ℓ = a -> -N*log(gamma(a)) + a*sum(Y2 - X*γ)-sum(exp.(Y2 - X*γ))

# a conditional
ca = a -> exp(sum(Y2 - X*γ)*a*(1+s))/gamma(a)
lca = a -> sum(Y2 - X*γ)*a*(1+s)-(N+r)*log(gamma(a))

fplot((0, 5), lca)
fplot!((1, 1.6), a -> (-N*digamma(a)-sum(Y2 - X*γ))/3000)
