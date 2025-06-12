using Distributions, StatsBase, ProgressBars
using FinData, Dates, DataFrames
using LinearAlgebra, Base.Threads, JLD2
using Plots, CustomPlots
# using Flux
include("../functions.jl")

function Format_TS_data(TS::AbstractVector{T}, P::Int, D::Int) where T <: Real
    if D != 0
        for _ in 1:D
            TS = diff(TS)
        end
    end

    X = hcat(
        repeat([1], length(TS)-P), 
        [TS[p:end-P+p-1] for p in P:-1:1]...
    )
    Y = TS[P+1:end]
    return X, Y
end

function BayesPost(X, Y; N::Int=50_000)
    n, m = size(X)

    # Priors
    β₀ = ones(m)*0
    λ = I(m)*0.5
    a = 0
    u = 0

    # Sample Posterior Distributions
    pΛ = Gamma(a + n/2, inv(u+0.5*(Y'*Y + β₀'*λ*β₀ - (X'*Y+λ*β₀)'*inv(X'*X+λ)*(X'*Y+λ*β₀))))
    Λ = rand(pΛ, N)
    β = inv(X'*X+λ)*(X'*Y+λ*β₀) .+ sqrt(inv(X'*X+λ))*randn(m, N) ./ sqrt.(Λ')

    # Posterior expectations
    μ = mean(β[1,:] ./ (1 .- sum(β[2:end,:]', dims = 2)[:])) # Mean value
    V = Λ .* (1 .- sum(β[2:end,:]', dims = 2).^2)[:] |> V -> mean((V[V .> 0]).^(-0.5)) # Variance
    return μ, V
end

# @doc"""
# # bot1

# Input a timeseries, outputs predictions for buy price, take profit.

# """
# function bot1(TS::Vector{T};) where T <: Real
    
# end


# ============================================================
# ==================   Load and Prep Data   ==================
# ============================================================

coins = ["AAVE", "AVAX", "BAT", "BCH", "BTC", "CRV", "DOGE", "ETH", "GRT", "LINK", "LTC", "MKR", "PEPE",  "SHIB", "SUSHI", "TRUMP", "UNI", "XRP", "XTZ"] # , "DOT","SOL", "YFI"

# Variables
tStep = Minute(5)
step = Day(1)
MW = Day(3)
trainPeriod = Week(4)
testPeriod = Week(2)
P = 2
a = 0.75

record = Dict{String, DataFrame}()
# @load "bot/data/bot2jld2" record


for coin in coins
    # if coin ∈ keys(record)
    #     println("Skip $coin")
    #     continue
    # end
    println("Simulating $coin")
    data = loadCryptoQuotes(coin)
    filter!(d -> d.bp != 0, data)
    transform!(data, :t => ByRow(t -> round(t, tStep, RoundDown)) => :t)

    minData = combine(groupby(data, :t),
        :bp => last => :bp,
        # :ap => mean => :ap,
    )
    fillMissing!(minData, step = tStep)
    transform!(minData, :bp => ByRow(log) => :X)

    barData = loadCryptoBars(coin)

    # ============================================================
    # ======================   Back Test   =======================
    # ============================================================

    buyOutcomes = Dict[]

    T = round(minData.t[1], Day, RoundDown)+trainPeriod+step
    MaxT = minData.t[end]-testPeriod-step

    for T in range(T, MaxT, step=step) |> ProgressBar

        TS = @view minData[T-trainPeriod .≤ minData.t .< T, :]

        X, Y = Format_TS_data(TS.X, P, 0)
        μ0, V0 = BayesPost(X, Y)

        TP = μ0
        BP = μ0 - a*V0

        TS2 = @view barData[T .< barData.t .≤ T+trainPeriod, :]

        BT = findfirst(TS2.l[TS2.t .≤ T+MW] .< exp(BP))
        if BT !== nothing
            WT = findfirst(TS2.h[BT:end] .> exp(TP))
            if WT !== nothing
                push!(buyOutcomes, Dict(
                    "startTime" => T,
                    "startPrice" => TS2.o[1],
                    "buyPrice" => exp(BP),
                    "takeProfit" => exp(TP),
                    "buyTime" => TS2.t[BT],
                    "ExecutionTime" => TS2.t[BT+WT],
                    "Outcome" => TS2.o[1] < exp(BP) ? exp(TP)/TS2.o[1] : exp(TP - BP)
                ))
            else
                push!(buyOutcomes, Dict(
                    "startTime" => T,
                    "startPrice" => TS2.o[1],
                    "buyPrice" => exp(BP),
                    "takeProfit" => exp(TP),
                    "buyTime" => TS2.t[BT],
                    "ExecutionTime" => TS2.t[end],
                    "Outcome" => TS2.o[1] < exp(BP) ? TS2.c[end]/TS2.o[1] : TS2.c[end]/exp(BP)
                ))
            end
        else
            push!(buyOutcomes, Dict(
                    "startTime" => T,
                    "startPrice" => TS2.o[1],
                    "buyPrice" => exp(BP),
                    "takeProfit" => exp(TP),
                    "buyTime" => DateTime(0),
                    "ExecutionTime" => DateTime(0),
                    "Outcome" => 1
                )
            )
        end
    end

    record[coin] = DataFrame(buyOutcomes)
end

@save "bot/data/bot2.jld2" record
# @load "bot/data/bot2.jld2" record

MakerFee = 0.25/100
TakerFee = 0.15/100
portion = 1/(10*5)

# [transform!(record[c], :startTime => ByRow(t -> round(t, Day, RoundDown)) => :startTime) for c in keys(record)]
# [sort!(record[c], :startTime) for c in keys(record)]

T0 = minimum([record[c].startTime[1] for c in  keys(record)])
MaxT = maximum([record[c].startTime[end] for c in keys(record)])

b0 = 0.8123960408246376
b1 = 0.20159852269506473

n = length(range(T, MaxT, step=step))
liquidity = 2_000
V = vcat(liquidity, zeros(n)) # Value
Liq = vcat(liquidity, zeros(n))

ActiveTrades = Dict[]
Trades = 0
failures = 0
# CompletedTrades = Dict[]

for (i, T) in enumerate(range(T, MaxT, step=step))
    # Complete trades which have finsihed before T
    V[i+1] = copy(V[i])
    for j in findall(x -> x["ExecutionTime"] <= T, ActiveTrades)
        X = ActiveTrades[j]
        Trades += 1
        if X["buyTime"] == DateTime(0)
            failures += 1
            liquidity += X["Value"]
        else
            V[i+1] += (X["Outcome"] - 1 - MakerFee - X["Outcome"]*TakerFee)*X["Value"]
            liquidity += (X["Outcome"] - MakerFee - X["Outcome"]*TakerFee)*X["Value"]
        end
    end
    filter!(x -> x["ExecutionTime"] > T, ActiveTrades)

    # Get best trading opportunities
    trades = [record[c][record[c].startTime .== T,:] for c in keys(record)]
    
    tv = [isempty(t) ? 0 : b0 + b1 * t.takeProfit[1]/t.buyPrice[1] for t in trades]
    idx = sortperm(tv, rev = true)
    # activate trade 
    for t in trades[idx[1:min(10, round(Int, length(tv[tv .!= 0])))]]
        if liquidity > 1
            push!(ActiveTrades, Dict(
                "Value" => min(V[i+1]*portion, liquidity),
                "ExecutionTime" => t.buyTime[1] == DateTime(0) ? t.startTime[1]+MW : t.ExecutionTime[1],
                "buyTime" => t.buyTime[1],
                "Outcome" => t.Outcome[1]
            ))
            liquidity -= min(V[i+1]*portion, liquidity)
        end
    end

    Liq[i+1] = liquidity
    # T += step
    # i += 1
end


plot(V, legend=false)
plot!(Liq)
plot(Liq./V)

failures/Trades

# ============================================================
# ====================   Decision Maker   ====================
# ============================================================

@load "bot/data/bot2.jld2" record

X = Matrix{Float64}(undef, 0, 3)
Y1 = Float64[]
Y2 = Float64[]

for c in keys(record)
    X = vcat(X, hcat(record[c].takeProfit ./ record[c].startPrice,
        record[c].buyPrice ./ record[c].startPrice,
        record[c].takeProfit ./ record[c].buyPrice)
    )

    append!(Y1, record[c].Outcome .!= 1.0)
    append!(Y2, record[c].Outcome)
end


X2 = X[Y1 .!= 0,:]
X2 = hcat(ones(size(X2,1)), X2)
Y2 = Y2[Y1 .!= 0]

β2 = inv(X2'*X2 + 10I(4))*X2'*Y2

model = Chain(
    Dense(3 => 5, tanh),
    Dense(5 => 3, relu),
    Dense(3 => 1, sigmoid),
)

loader = Flux.DataLoader((X', Y1'), batchsize=64, shuffle=true)
opt = Flux.setup(Adam(), model)

losses = Float64[]
for epoch in 1:1_000 |> ProgressBar
    for data in loader
        # Unpack batch of data, and move to GPU:
        x, y = data
        loss, grads = Flux.withgradient(model) do m
            # Evaluate model and loss inside gradient context:
            y_hat = m(x)
            Flux.crossentropy(y_hat, y)
        end
        Flux.update!(opt, model, grads[1])
        push!(losses, loss)  # logging, outside gradient context
    end
end

plot(losses)

model(X')

D = DataFrame(Y=Y1, X1 = X[:,1], X2 = X[:,2], X3=X[:,3])

CSV.write("data/predTask.csv", D)

