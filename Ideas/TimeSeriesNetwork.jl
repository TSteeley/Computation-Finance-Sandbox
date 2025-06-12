using Plots, CustomPlots, ProgressBars, Distributions
using FinData, Dates, DataFrames, ProgressBars
using LinearAlgebra, Flux
using SpecialFunctions
include("../functions.jl")

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

trainPeriod = Week(3)
testPeriod = Week(1)

r = sample(minData.t[1], minData.t[end]-(trainPeriod+testPeriod+Week(1)), step = tStep)
train = filter(x -> r .≤ x.t .< r+trainPeriod, minData)
test = filter(x -> r+trainPeriod .≤ x.t .< r+trainPeriod+testPeriod, minData)

# ============================================================
# ====================   Neural Network   ====================
# ============================================================

LB = 5 # Look behind
LA = 1 # Look ahead

X = hcat([Float32.(train.X[i-LB+1:i]) for i in LB:length(train.X)-LA]...)
Y = hcat([Float32.(train.X[i+1:LA+i]) for i in LB:length(train.X)-LA]...)

function loss(y, yhat)
    return -mean(loglikelihood.(Normal.(yhat[1,:],exp.(yhat[2,:])),y'))
end

model = Flux.Chain(
    Dense(LB => 5, relu),
    Dense(5 => 10, relu),
    Dense(10 => 5, relu),
    Dense(5 => 2),
)

loader = Flux.DataLoader((X, Y), batchsize = 64, shuffle = true)
opt = Flux.setup(Adam(), model)

losses = []
for epoch in 1:1000 |> ProgressBar
    l = 0
    for data in loader
        x, y = data
        Loss, grads = Flux.withgradient(model) do m
            # Flux.mse(y, m(x))
            loss(y, m(x))
        end
        Flux.update!(opt, model, grads[1])
        l += Loss*length(x)
    end
    push!(losses, l/size(X,2))
end

plot(losses[10:end])

plot(train.t, train.X, label = "")
plot!(test.t, test.X, label = "")

y = zeros(Float32, length(test.t))
y[1:LB] .= test.X[1:LB]

for i in LB:length(y)-1
    # y[i+1] = model(y[i-LB+1:i])[1]
    μ, σ = model(y[i-LB+1:i])
    y[i+1] = rand(Normal(μ,exp(σ)))
end
plot!(test.t, y, label = "")


X = [1:4;]

[1 0 ; 1 1 ; 1 1 ; 0 1]