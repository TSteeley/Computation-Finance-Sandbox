#=
    bot_nh.jl

    Null hypothesis bot. Makes decisions totally randomly, baseline for any other bot.

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

# ============================================================
# ===================   Hyper Parameters   ===================
# ============================================================

# Valid Coins
coins = ["AAVE", "AVAX", "BAT", "BCH", "BTC", "CRV", "DOGE", "ETH", "GRT", "LINK", "LTC", "MKR", "SUSHI", "TRUMP", "UNI", "XRP", "XTZ"] # , "DOT", "SOL", "YFI", "PEPE",  "SHIB"
tStep = Minute(5) # Chunking of quotes
step = Hour(6) # How often esimates are made on data
MW = Day(3) # Maximum wait for an order to execute
trainPeriod = Week(4) # Window size
testPeriod = Week(2) # Time a trade has to completely finish
P = 2 # How long back AR looks
a = 0.75

# ============================================================
# ====================   Compute Values   ====================
# ============================================================

# Store predictions; dictionary of DataFrames
# Either start from scratch or continue from already done predictions
preds = Dict{String, DataFrame}()
# @load "bot/data/bot_nh.jld2" preds

for coin in coins
    # if coin ∈ keys(preds)
    #     println("Skip $coin")
    #     continue
    # end
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

    for T in range(T0, MaxT, step=step) |> ProgressBar
        TS = @view minData[T-trainPeriod .≤ minData.t .< T,:]

        push!(predictions, Dict(
                "t" => T,
                "μ0" => TS.X[end]*(0.8+rand()*0.4),
                "V0" => TS.X[end]*0.2*rand(),
            )
        )
    end

    preds[coin] = DataFrame(predictions)
end

@save "bot/data/bot_nh.jld2" preds

# ============================================================
# ================   Compute Trade Outcomes   ================
# ============================================================

@load "bot/data/bot_nh.jld2" preds
trades = Dict{String, DataFrame}()

for coin in coins
    println("Simulating $coin")

    barData = loadCryptoBars(coin)

    # ============================================================
    # ======================   Back Test   =======================
    # ============================================================

    buyOutcomes = Dict[]
    T0 = round(barData.t[1], Day, RoundDown)+trainPeriod+step
    MaxT = barData.t[end]-testPeriod

    for T in preds[coin].t |> ProgressBar

        BP = log(barData[barData.t .<= T, :o][end] * (0.95+0.1*rand()))
        TP = log(exp(BP)*(1+0.1*rand()))

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

    trades[coin] = DataFrame(buyOutcomes)
end

@save "bot/data/bot_nh.jld2" trades

# ============================================================
# ===================   Evaluate Strategy   ==================
# ============================================================

@load "bot/data/bot_nh.jld2" trades

MakerFee = 0.25/100
TakerFee = 0.15/100
portion = 1/(10*5)

T0 = minimum([trades[c].startTime[1] for c in  keys(trades)])
MaxT = maximum([trades[c].startTime[end] for c in keys(trades)])

n = length(range(T, MaxT, step=step))
liquidity = 2_000
V = vcat(liquidity, zeros(n)) # Value
Liq = vcat(liquidity, zeros(n))

ActiveTrades = Dict[]
Trades = 0
failures = 0
# CompletedTrades = Dict[]

b0 = 0.8123960408246376
b1 = 0.20159852269506473

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
    currentTrades = [trades[c][trades[c].startTime .== T,:] for c in keys(trades)]
    
    tv = [isempty(t) ? 0 : b0 + b1 * t.takeProfit[1]/t.buyPrice[1] for t in currentTrades]
    idx = sortperm(tv, rev = true)
    # activate trade 
    for t in currentTrades[idx[1:min(10, round(Int, length(tv[tv .!= 0])))]]
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
# ========   Guess when trade will fail to execute   =========
# ============================================================

