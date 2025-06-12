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

function ±(x, y)
    return x .+ [1, -1] .* y
end
function ∓(x, y)
    return x .+ [-1, 1] .* y
end

# ============================================================
# ==================   Load and Prep Data   ==================
# ============================================================

tStep = Minute(5)
data = loadCryptoQuotes("BTC")
filter!(d -> d.as != 0, data)
transform!(data, :t => ByRow(t -> round(t, tStep, RoundDown)) => :t)

minData = combine(groupby(data, :t),
    :bp => last => :bp,
    # :ap => mean => :ap,
)
fillMissing!(minData, step = tStep)
transform!(minData, :bp => ByRow(log) => :X)

trainPeriod = Week(4)
testPeriod = Week(2)

r = sample(minData.t[1], minData.t[end]-(trainPeriod+testPeriod+Week(1)), step = tStep)
train = filter(x -> r .≤ x.t .< r+trainPeriod, minData)
test = filter(x -> r+trainPeriod .≤ x.t .< r+trainPeriod+testPeriod, minData)

# ============================================================
# ======================   ARIMA model   =====================
# ============================================================

X, Y = Format_TS_data(train.X, P, 0)
a = inv(X'*X)*X'*Y |> x -> x[1]/(1-sum(x[2:end]))
X, Y = Format_TS_data(train.X, P, 1)
b = inv(X'*X)*X'*Y |> x -> x[1]/(1-sum(x[2:end]))
X, Y = Format_TS_data(train.X, P, 2)
c = inv(X'*X)*X'*Y |> x -> x[1]/(1-sum(x[2:end]))

# λ = exp.(-(length(Y):-1:1)./length(Y))
# a = inv((X.*λ)'*X)*(X.*λ)'*Y |> x -> x[1]/(1-sum(x[2:end]))
# λ = exp.(-(length(Y):-1:1)./length(Y))
# b = inv((X.*λ)'*X)*(X.*λ)'*Y |> x -> x[1]/(1-sum(x[2:end]))


plot(1:length(train.t), train.X, label = "")
plot!(length(train.t).+(1:length(test.t)), test.X, label = "")
fplot!(1:length(train.t)+length(test.t), x -> a, label = "")
fplot!(1:length(train.t)+length(test.t), x -> a+b*x - length(train.t)*b, label = "")
# hline!([mean(train.X)])

# ============================================================
# =======================   Moving AR   ======================
# ============================================================

minData[!,"AR"] = zeros(length(minData.t))
minData[!,"ARI"] = zeros(length(minData.t))

LB = Week(3)
P = 5

n = Int(LB / tStep)

for i in n:length(minData.t)
    X = hcat(
        ones(n-P), 
        [minData.X[p+i-n:p+i-P-1] for p in 1:P]...
    )
    Y = minData.X[i-n+P+1:i]
    minData[i,:AR] = inv(X'*X)*X'*Y |> β -> β[1]/(1-sum(β[2:end]))
end

pData = minData[100_000:170_000,:]

plot(pData.t, pData.X, ylims=[10, 12])
plot!(pData.t, pData.AR)

# ============================================================
# =======================   Var of AR   ======================
# ============================================================


# β = inv(X'*X)*X'*Y
# μ = β[1]/(1-sum(β[2:end]))
# Σ = (Y - X*β)'*(Y - X*β) / n
# σ = √Σ

# N = 100_000
# y = ones(N)*mean(train.X)

# for i in P+1:N
#     y[i] = [1, y[i-P:i-1]...]'*β + randn()*σ
# end

# V = mean((y .- mean(y)).^2)

# p = plot(legend = false)
# plot!(p, y)
# hline!(p, [mean(y)])
# hline!(p, [β[1]/(1-sum(β[2:end]))])
# hline!(p, quantile(y, [0.025, 0.975]))
# hline!(p, β[1]/(1-sum(β[2:end])) ∓ 1.96 * sqrt(Σ/(1-sum(β[2:end])^2)))

function BayesPost(X, Y)
    β₀ = ones(m)*0
    λ = I(m)*0.5
    a = 0
    u = 0

    pΛ = Gamma(a + n/2, inv(u+0.5*(Y'*Y + β₀'*λ*β₀ - (X'*Y+λ*β₀)'*inv(X'*X+λ)*(X'*Y+λ*β₀))))
    pβ = Λ -> MvNormal(inv(X'*X+λ)*(X'*Y+λ*β₀), inv(Λ*(X'*X+λ)) |> Hermitian)

    N = 50_000

    Λ1 = rand(pΛ, N)
    β1 = hcat(rand.(pβ.(Λ1))...)

    μ = β1[1,:] ./ (1 .- sum(β1[2:end,:]', dims = 2)[:])
    V = 1 ./(Λ1 .* (1 .- sum(β1[2:end,:]', dims = 2).^2)[:])
    return μ, V
end

r = sample(minData.t[1], minData.t[end]-(trainPeriod+testPeriod+Week(1)), step = tStep)
train = filter(x -> r .≤ x.t .< r+trainPeriod, minData)
test = filter(x -> r+trainPeriod .≤ x.t .< r+trainPeriod+testPeriod, minData)

P = 2
X, Y = Format_TS_data(train.X, P, 0)
n, m = size(X)

μ0, V0 = BayesPost(Format_TS_data(train.X, P, 0)...)
μ1, V1 = BayesPost(Format_TS_data(train.X, P, 1)...)

p1 = plot(legend = false)
plot!(p1, train.t, train.X)
plot!(p1, test.t, test.X)
hline!([mean(μ0)])
hline!(p1, mean(μ0) ∓ mean(sqrt.(V0[V0.> 0])))
hline!(p1, mean(μ0) ∓ 0.5mean(sqrt.(V0[V0.> 0])))
plot!(p1, test.t, mean(μ0) .+[0:length(test.t)-1;]*mean(μ1))

mean(μ1) ∓ mean(sqrt.(V1[V1.> 0]))

μ = exp(mean(μ0))
L, U = exp.(mean(μ0) ∓ mean(sqrt.(V0[V0.> 0])))

train.bp[end]/μ

# ============================================================
# =====================   Reject Params   ====================
# ============================================================

r = sample(minData.t[1], minData.t[end]-(trainPeriod+testPeriod+Week(1)), step = tStep)
train = filter(x -> r .≤ x.t .< r+trainPeriod, minData)
test = filter(x -> r+trainPeriod .≤ x.t .< r+trainPeriod+testPeriod, minData)

P = 2
X, Y = Format_TS_data(train.X, P, 0)
n, m = size(X)

β₀ = ones(m)*0
λ = I(m)*1
a = 0
u = 0

pΛ = Gamma(a + n/2, inv(u+0.5*(Y'*Y + β₀'*λ*β₀ - (X'*Y+λ*β₀)'*inv(X'*X+λ)*(X'*Y+λ*β₀))))
pβ = Λ -> MvNormal(inv(X'*X+λ)*(X'*Y+λ*β₀), inv(Λ*(X'*X+λ)) |> Hermitian)

N = 10_000

Λ1 = rand(pΛ, N)
β1 = hcat(rand.(pβ.(Λ1))...)

μ = β1[1,:] ./ (1 .- sum(β1[2:end,:]', dims = 2)[:])
V = 1 ./(Λ1 .* (1 .- sum(β1[2:end,:]', dims = 2).^2)[:])

sqrt.(1 ./ (Λ1 .*(1 .-sum(β1[2:end,:], dims = 1)[:].^2))) |> mean

histogram(β1[21,:])

scatter(mean(β1, dims = 2)[:], yerr = hcat([quantile(x, [0.025, 0.975]) for x in eachrow(β1)]...)' .-mean(β1, dims = 2)[:], lw = 1)

[quantile(x, [0.025, 0.975]) for x in eachrow(β1)]