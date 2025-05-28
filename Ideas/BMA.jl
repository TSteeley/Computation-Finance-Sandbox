using Distributions, StatsBase
using Plots, CustomPlots
using FinData, Dates, DataFrames
using LinearAlgebra

data = loadCryptoQuotes("BTC")
filter!(x -> x.ap != 0, data)
transform!(data, :t => ByRow(t -> round(t, Dates.Minute, RoundDown)) => :t)

minData = combine(groupby(data, :t),
    :bp => last => :bp,
    :ap => last => :ap,
)
transform!(minData,
    :t => ByRow(t -> Dates.Minute(t - minData.t[1]).value) => :T
)

# ============================================================
# =====================   Format Data   ======================
# ============================================================

U = findlast(minData.t .< minData.t[end] - Dates.Week(4))

r = rand(1:U)
train = filter(x -> r .≤ x.T .< r+20160, Y)
test = filter(x -> r+20160 .≤ x.T .< r+30240, Y)

plot(train.t, train.ap)
plot!(test.t, test.ap)

idx = findall(diff(train.T) .== 1)
Y = diff(log.(train.ap))[idx]
X = hcat(train.T[idx] .-train.T[1], train.ap[idx])

# histogram(Y)
# ξ = fit(Normal, Y)
# fplot!([-0.02, 0.02], x -> pdf(ξ, x))

# ============================================================
# ======================   Fit Models   ======================
# ============================================================

function OrderN(X::AbstractMatrix{Float64}, Y::Vector{Float64}; o)
    X = hcat(
        ones(size(X,1)),
        [X[:,1].^i for i in 1:o[1]]...,
        [X[:,2].^i for i in 1:o[2]]...,
    )
    
    β = inv(X'*X)*X'*Y
    βσ = inv(X'*X)*X'*(Y - X*β).^2
    
    return x -> Normal(
        (hcat(1,x[1].^[1:o[1];]', x[2].^[1:o[2];]')*β)[1], 
        sqrt((hcat(1,x[1].^[1:o[1];]', x[2].^[1:o[2];]')*βσ)[1])
    )
end

function OrderN2(X::AbstractMatrix{Float64}, Y::Vector{Float64}; o)
    X = hcat(
        ones(size(X,1)),
        [X[:,1].^i for i in 1:o[1]]...,
        [X[:,2].^i for i in 1:o[2]]...,
    )
    
    β = inv(X'*X)*X'*Y.^2
    
    return x -> Normal(
        0, 
        sqrt((hcat(1,x[1].^[1:o[1];]', x[2].^[1:o[2];]')*β)[1])
    )
end

order = [3, 3]
M = OrderN(X, Y, o = order)

xi = [M(x) for x in eachrow(X[rand(1:8591, 5000),:])]

histogram(Y)
fplot!([-0.02, 0.02], x -> mean(pdf.(xi, x)))
fplot!([-0.02, 0.02], x ->pdf(ξ, x))

X = range(0, 1, length = 1001)
Y = randn(1001) .* sqrt.(X)

plot(cumsum(Y))

mean(Y.^2)

X = hcat(ones(1001), X)

β = inv(X'*X)*X'*Y.^2

