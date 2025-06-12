#=
    preComputeBot2.jl

    By far the longest parts of my tests is the Bayesian part. Here I just pre-compute that part, then hopefully I can test parameters at a much faster rate.

    Author: Thomas P. Steele
=#

# ============================================================
# ===================   Helper Functions   ===================
# ============================================================

using Distributions, StatsBase, ProgressBars
using FinData, Dates, DataFrames
using LinearAlgebra, Base.Threads, JLD2
using Plots, CustomPlots
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
    # .! isnan(V) || error()
    return μ, V
end

function arPost(X, Y; N::Int=10_000)
    n, m = size(X)
    # Priors
    β₀ = ones(m)*0
    λ = I(m)*0.1
    a = 0
    u = 0

    pΛ = Gamma(a + n/2, inv(u+0.5*(Y'*Y + β₀'*λ*β₀ - (X'*Y+λ*β₀)'*inv(X'*X+λ)*(X'*Y+λ*β₀))))

    mβ = inv(X'*X+λ)*(X'*Y+λ*β₀)
    σβ = sqrt(inv(X'*X+λ))

    y = zeros(N)
    yn = [1 ; ones(P)*mβ[1]/(1-sum(mβ[2:end]))]

    # Transition matrix
    A = zeros(P+1,P+1)
    A[1,1] = 1
    A[3:end,2:end-1] += I(P-1)

    # Burn in
    for _ in 1:100
        σ = 1/sqrt(rand(pΛ))
        β = mβ + σβ*randn(m)*σ
        A[2,:] .= β
        yn = A*yn
        yn[2] += σ*randn()
    end

    for i in 1:N
        σ = 1/sqrt(rand(pΛ))
        β = mβ + σβ*randn(m)*σ
        A[2,:] .= β
        yn = A*yn
        yn[2] += σ*randn()
        y[i] = yn[2]
    end

    return mean(y), std(y)
end

# ============================================================
# ===================   Hyper Parameters   ===================
# ============================================================

# Valid Coins
coins = ["AAVE", "AVAX", "BAT", "BCH", "BTC", "CRV", "DOGE", "ETH", "GRT", "LINK", "LTC", "MKR", "SUSHI", "TRUMP", "UNI", "XRP", "XTZ"] # , "DOT", "SOL", "YFI", "PEPE", "SHIB"
tStep = Minute(5) # Chunking of quotes
step = Hour(6) # How often esimates are made on data
MW = Day(3) # Maximum wait for an order to execute
trainPeriod = Week(4) # Window size
testPeriod = Week(2) # Time a trade has to completely finish
P = 5 # How long back AR looks
# a = 0.75

# ============================================================
# ====================   Compute Values   ====================
# ============================================================

# Store predictions; dictionary of DataFrames
# Either start from scratch or continue from already done predictions
preds = Dict{String, DataFrame}()
@load "bot/data/bot1.jld2" preds

for coin in coins
    if coin ∈ keys(preds)
        println("Skip $coin")
        continue
    end
    println("$coin")
    data = loadCryptoQuotes(coin)
    filter!(d -> d.bp != 0, data)
    transform!(data, :t => ByRow(t -> round(t, tStep, RoundDown)) => :t)

    minData = combine(groupby(data, :t),
        :bp => last => :bp,
        # :ap => mean => :ap,
    )
    fillMissing!(minData, step = tStep)
    transform!(minData, :bp => ByRow(log) => :X)

    # ============================================================
    # ======================   Back Test   =======================
    # ============================================================

    predictions = Dict[]

    T0 = round(minData.t[1], Day, RoundDown)+trainPeriod+step
    MaxT = minData.t[end]-testPeriod

    for (i, T) in enumerate(range(T0, MaxT, step=step)) |> ProgressBar
        TS = @view minData[T-trainPeriod .≤ minData.t .< T,:]

        X, Y = Format_TS_data(TS.X, P, 0)
        μ0, V0 = arPost(X, Y)
        X, Y = Format_TS_data(TS.X, P, 1)
        μ1, V1 = arPost(X, Y)
        X, Y = Format_TS_data(TS.X, P, 2)
        μ2, V2 = arPost(X, Y)

        if any(isnan.([μ0, V0, μ1, V1, μ2, V2]))
            println("Fail; coin=$coin, T=$T, i=$i")
            break
        end

        push!(predictions, Dict(
                "t" => T,
                "μ0" => μ0,
                "V0" => V0,
                "μ1" => μ1,
                "V1" => V1,
                "μ2" => μ2,
                "V2" => V2,
                "c"  => TS.X[end]
            )
        )
    end

    preds[coin] = DataFrame(predictions)
end

@save "bot/data/bot1.jld2" preds

# coin = coins[3]
# [count(isnan.(Matrix(select(preds[c], Not(:t))))) for c in coins] |> sum

# ============================================================
# ================   Compute Trade Outcomes   ================
# ============================================================

@load "bot/data/bot_weights.jld2" β
@load "bot/data/bot1.jld2" preds
trades = Dict{String, DataFrame}()

for coin in coins
    println("Simulating $coin")

    barData = loadCryptoBars(coin)

    # ============================================================
    # ======================   Back Test   =======================
    # ============================================================

    buyOutcomes = Dict[]

    for T in preds[coin].t |> ProgressBar
        TP, BP = hcat(1, Matrix(select(preds[coin][preds[coin].t .== T, :], [:μ0,:V0,:μ1,:V1,:μ2,:V2,:c])))*β
        # TP = preds[coin]

        # TP = preds[coin][preds[coin].t .== T, "μ0"][1]
        # BP = TP - a*preds[coin][preds[coin].t .== T, "V0"][1]

        TS2 = @view barData[T .< barData.t .≤ T+testPeriod, :]

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
                    "ExecutionTime" => TS2.t[BT+WT-1],
                    "ExecutionPrice" => exp(TP),
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
                    "ExecutionPrice" => TS2.c[end],
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
                    "ExecutionTime" => T+MW,
                    "ExecutionPrice" => 0,
                    "Outcome" => 1
                )
            )
        end
    end

    trades[coin] = DataFrame(buyOutcomes)
end

@save "bot/data/bot1.jld2" preds trades

c = "AAVE"
O = vcat([trades[c].Outcome for c in keys(trades)]...)

histogram(O[O .!= 1])

mean(O .== 1)

# ============================================================
# ===================   Evaluate Strategy   ==================
# ============================================================

@load "bot/data/bot1.jld2" preds trades

# coins[13]
# coins[14]
# delete!(trades, coins[13])
# delete!(trades, coins[14])

MakerFee = 0.25/100
TakerFee = 0.15/100
portion = 1/(12*12*1.2)

T0 = minimum([trades[c].startTime[1] for c in  keys(trades)])
MaxT = maximum([trades[c].startTime[end] for c in keys(trades)])

n = length(range(T0, MaxT, step=step))
liquidity = 2_000
V = vcat(liquidity, zeros(n)) # Value
Liq = vcat(liquidity, zeros(n))

ActiveTrades = Dict[]
Trades = 0
failures = 0
# CompletedTrades = Dict[]

b0 = 0.8123960408246376
b1 = 0.20159852269506473

for (i, T) in enumerate(range(T0, MaxT, step=step))
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
    currentTrades = [trades[c][trades[c].startTime .== T,:] for c in keys(trades)]
    filter!(x -> .! isempty(x), currentTrades)
    
    tv = [isempty(t) ? 0 : t.takeProfit[1]/t.buyPrice[1] for t in currentTrades]
    # tv = [isempty(t) ? 0 : t.startPrice[1]/t.buyPrice[1] for t in currentTrades]
    # tv = [isempty(t) ? 0 : b0 + b1 * t.takeProfit[1]/t.buyPrice[1] for t in currentTrades]

    idx = sortperm(tv, rev = true)
    # idx = sortperm(tv)
    # activate trade 
    for t in currentTrades[idx[1:min(10, round(Int, length(tv[tv .!= 0])))]]
        if liquidity > 1
            push!(ActiveTrades, Dict(
                "Value" => min(V[i+1]*portion, liquidity),
                "ExecutionTime" => t.ExecutionTime[1],
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
# ===================   Rolling Strategy   ===================
# ============================================================

@load "bot/data/bot1.jld2" preds trades

MakerFee = 0.25/100
TakerFee = 0.15/100
portion = 1/(5*12)

T0 = minimum([trades[c].startTime[1] for c in  keys(trades)])
MaxT = maximum([trades[c].startTime[end] for c in keys(trades)])
n = length(range(T0, MaxT, step=step))

liquidity = 2_000
V = [liquidity ; zeros(n)] # Value
Liq = [liquidity ; zeros(n)]

ActiveTrades = Dict[]
ActiveOrders = Dict[]
CompletedTrades = Dict[]
Trades = 0
failures = 0

b0 = 0.8123960408246376
b1 = 0.20159852269506473

for (i, T) in enumerate(range(T0, MaxT, step=step))
    V[i+1] = copy(V[i])
    # Complete trades which have finsihed before T
    for j in findall(x -> x["ExecutionTime"] <= T, ActiveTrades)
        X = ActiveTrades[j]
        Trades += 1
        V[i+1] += X["Units"]*(1-TakerFee)*X["ExecutionPrice"] - X["Value"]
        liquidity += X["Units"]*(1-TakerFee)*X["ExecutionPrice"]
        push!(CompletedTrades, Dict(
            "coin" => X["coin"],
            "Value" => X["Value"],
            "Units" => X["Units"],
            "ExecutionTime" => X["ExecutionTime"],
            "takeProfit" => X["takeProfit"],
            "ExecutionPrice" => X["ExecutionPrice"],
            "Gain" => (X["Units"]*(1-TakerFee)*X["ExecutionPrice"] - X["Value"])/X["Value"]
        ))
    end
    filter!(x -> x["ExecutionTime"] > T, ActiveTrades)

    # Check if orders executed
    for t in findall(x -> (x["buyTime"] != DateTime(0)) && (x["buyTime"] <= T), ActiveOrders)
        X = ActiveOrders[t]
        push!(ActiveTrades,Dict(
            "coin" => X["coin"],
            "Value" => X["Value"]*(1-TakerFee),
            "Units" => X["Value"]*(1-TakerFee)/X["startPrice"],
            "ExecutionTime" => X["ExecutionTime"],
            "ExecutionPrice" => X["ExecutionPrice"],
            "takeProfit" => X["takeProfit"],
        ))
    end
    filter!(x -> .! ((x["buyTime"] != DateTime(0)) && (x["buyTime"] <= T)), ActiveOrders)
    
    # Consolidate Orders
    for c in coins
        # Find all trades on coin
        idx = findall(x -> x["coin"] == c, ActiveTrades)
        if .! isempty(idx)
            # Newest trade
            _, newest = findmax(x -> x["ExecutionTime"], ActiveTrades[idx])
            toKill = Int[]
            for j in idx[idx .!= idx[newest]]
                if ActiveTrades[j]["takeProfit"] < ActiveTrades[idx[newest]]["takeProfit"]
                    ActiveTrades[idx[newest]]["Units"] += ActiveTrades[j]["Units"]
                    ActiveTrades[idx[newest]]["Value"] += ActiveTrades[j]["Value"]
                    push!(toKill, j)
                end
            end
            deleteat!(ActiveTrades, toKill)
        end
    end

    # Delete finished orders
    for j in findall(x -> (x["buyTime"] == DateTime(0)) && (x["buyTime"] <= T) && (x["ExecutionTime"] <= T), ActiveOrders)
        X = ActiveOrders[j]
        liquidity += X["Value"]
        failures += 1
    end
    filter!(x -> x["ExecutionTime"] > T, ActiveOrders)

    # Get trade predictions
    currentTrades = [(trades[c][trades[c].startTime .== T,:], c) for c in keys(trades)]

    # Sort by best
    tv = [isempty(t[1]) ? 0 : t[1].takeProfit[1]/t[1].buyPrice[1] for t in currentTrades]
    # tv = [isempty(t[1]) ? 0 : b0 + b1 * t[1].takeProfit[1]/t[1].buyPrice[1] for t in currentTrades]
    idx = sortperm(tv, rev = true)

    # Place Orders
    for (t,c) in currentTrades[idx[1:min(5, round(Int, length(tv[tv .!= 0])))]]
        if liquidity > 1
            if t[1,"startPrice"] < t[1,"buyPrice"] # Market order
                push!(ActiveTrades, Dict(
                    "coin" => c,
                    "Value" => min(V[i+1]*portion, liquidity)*(1-MakerFee),
                    "Units" => min(V[i+1]*portion, liquidity)*(1-MakerFee)/t.startPrice[1],
                    "ExecutionTime" => t.ExecutionTime[1],
                    "ExecutionPrice" => t.ExecutionPrice[1],
                    "takeProfit" => t.takeProfit[1],
                ))
            else # Limit order
                push!(ActiveOrders, Dict(
                    "coin" => c,
                    "Value" => min(V[i+1]*portion, liquidity),
                    "ExecutionTime" => t.ExecutionTime[1],
                    "ExecutionPrice" => t.ExecutionPrice[1],
                    "startPrice" => t.startPrice[1],
                    "startTime" => t.startTime[1],
                    "buyTime" => t.buyTime[1],
                    "buyPrice" => t.buyPrice[1],
                    "takeProfit" => t.takeProfit[1],
                    "Outcome" => t.Outcome[1]
                ))
            end
            liquidity -= min(V[i+1]*portion, liquidity)
        end
    end

    Liq[i+1] = liquidity
    # T += step
    # i += 1
    # println(i)
end

plot(V, legend=false)
plot!(Liq)
plot(Liq./V)

failures/Trades

G = [x["Gain"] for x in CompletedTrades]

# ============================================================
# ========   Guess when trade will fail to execute   =========
# ============================================================

# Best Possible Trade Outcomes
BPTO = Dict{String, DataFrame}()

T0 = minimum([preds[c].t[1] for c in coins])
MaxT = T0 + Year(1)

for coin in coins
    println("Simulating $coin")

    barData = loadCryptoBars(coin)
    bestBuys = Dict[]

    for T in preds[coin][preds[coin].t .< MaxT,:].t |> ProgressBar
        TS2 = @view barData[T .< barData.t .≤ T+testPeriod, :]

        BBP = quantile(TS2.l[TS2.t .≤ T+MW], 0.2)
        BBT = findfirst(TS2.l .≤ BBP)
        BTP = quantile(TS2.h[BBT:end], 0.8)
        # BBP, BBT = findmin(TS2.l[TS2.t .≤ T+MW])
        # BTP = maximum(TS2.h[BBT:end])

        push!(bestBuys, Dict("BBP" => BBP, "BTP" => BTP, "t"=>T))
    end

    if length(preds[coin][preds[coin].t .< MaxT,:].t) != 0
        BPTO[coin] = DataFrame(bestBuys)
    end

end

YTP = log.(vcat([BPTO[c].BTP for c in keys(BPTO)]...))
YBP = log.(vcat([BPTO[c].BBP for c in keys(BPTO)]...))
X = vcat([Matrix(select(preds[c][preds[c].t .< MaxT,:], [:μ0,:V0,:μ1,:V1,:μ2,:V2,:c])) for c in keys(BPTO)]...)
X = hcat(ones(size(X,1)), X)

count(isnan.(X))

βTP = inv(X'*X)*X'*YTP
βBP = inv(X'*X)*X'*YBP

YBP - X*βBP
YTP - X*βTP

β = [βTP βBP]

@save "bot/data/bot_weights.jld2" β
# @save "bot/data/bot_weights2.jld2" β

Y - X*β
c = "AAVE"
a = Matrix(select(preds[c], Not(:t)))
findall(isnan.(a[:,4]))