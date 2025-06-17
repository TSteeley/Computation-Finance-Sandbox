#=
    bot1.jl

    This file contains all of the code for bot1 where I was testing it before my first test deployment. 

    Aspects of the process do not need to be run sequentially, so I do not. This helps speed up iterations and bug finding.

    Breakdow of the sections in this file,
    1. Load and prep dependencies and functions
    2. Hyper parameters for bot1
    3. Precompute predictions for all coins at all time steps
    4. Create a predictor for the best take-profit and buy-price to use for each trade
    5. Use predictor (step 4) for each coin at each time-step and see how trades play out
    6. Backtest. Step through time making decisions as you would do in a real scenario to see how bot performs.

    Author: Thomas P. Steele
=#

using Distributions, StatsBase, ProgressBars
using FinData, Dates, DataFrames
using LinearAlgebra, Base.Threads, JLD2
using Plots, CustomPlots
include("../functions.jl")

@doc raw"""
    Format_TS_data(TS::AbstractVector{<:Real}, P::Int, D::Int)

Formats a time series (TS) for bot1. Returns X, Y
```math
TS = [x_1,x_2,\dots,x_n]^T
  X = [
      1  x_P       x_{P-1}  ⋯  x_1
      1  x_{P+1}   x_P      ⋯  x_2
      ⋮   ⋮         ⋮        ⋮   ⋮
  ]
  Y = [x_{P+1},x_{P+2},⋯,x_{n}]^T
```
...
# Arguments
- `P::Int`: number of previous steps to include in X
- `D::Int`: take the difference of each step in X D times
...
"""
function Format_TS_data(TS::AbstractVector{<:Real}, P::Int, D::Int)
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

@doc"""
    arPost(X::AbstractMatrix{<:Real}, Y::AbstractVector{<:Real}; N::Int=10_000)
Fits an autoregressive model with data X and Y. Returns the posterior expectation and standard deviation of fit time series.
...
# Arguments
- `N::Int=10_000`: number of steps in monte-carlo simulation of time series. Used for estimating time series expectation and standard deviation.
...
"""
function arPost(X::AbstractMatrix{<:Real}, Y::AbstractVector{<:Real}; N::Int=10_000)
    n, m = size(X)
    # Priors
    β₀ = ones(m)*0
    λ = I(m)*0.1
    a = 0
    u = 0

    pΛ = Gamma(a + n/2, inv(u+0.5*(Y'*Y + β₀'*λ*β₀ - (Y'*X+β₀'*λ)*inv(X'*X+λ)*(X'*Y+λ*β₀))))

    mβ = inv(X'*X+λ)*(X'*Y+λ*β₀)
    σβ = sqrt(inv(X'*X+λ))

    y = zeros(N)
    yn = [1 ; ones(m-1)*mβ[1]/(1-sum(mβ[2:end]))]

    # Transition matrix
    A = zeros(m,m)
    A[1,1] = 1
    A[3:end,2:end-1] += I(m-2)

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

    if abs((mean(y[N÷2+1:end]) - mean(y[1:N÷2]))/std(y[1:N÷2])) < 3
        return mean(y), std(y)
    else
        return NaN, NaN
    end
end

function arPost_v2(ts::AbstractVector{<:Real}, P::Int, D::Int; N::Int=10_000)
    n = length(ts)-P-D
    m = P+1
    # Take difference D times
    if D != 0
        for _ in 1:D
            ts = diff(ts)
        end
    end

    X = hcat(
        repeat([1], length(ts)-P), 
        [view(ts,p:n+p-1) for p in P:-1:1]...
    )
    Y = @view ts[P+1:end]

    # Priors
    β₀ = ones(m)*0
    λ = I(m)*0.5
    a = 0
    u = 0

    pΛ = Gamma(a + n/2, inv(u+0.5*(Y'*Y + β₀'*λ*β₀ - (Y'*X+β₀'*λ)*inv(X'*X+λ)*(X'*Y+λ*β₀))))

    mβ = inv(X'*X+λ)*(X'*Y+λ*β₀)
    σβ = sqrt(inv(X'*X+λ))

    y = zeros(N)
    yn = [1 ; ones(m-1)*mβ[1]/(1-sum(mβ[2:end]))]

    # Transition matrix
    A = zeros(m,m)
    A[1,1] = 1
    A[3:end,2:end-1] += I(m-2)

    # Burn in
    for _ in 1:100
        σp = 1/sqrt(rand(pΛ))
        β = mβ + σβ*randn(m)*σp
        A[2,:] .= β
        yn = A*yn
        yn[2] += σp*randn()
    end

    for i in 1:N
        σp = 1/sqrt(rand(pΛ))
        β = mβ + σβ*randn(m)*σp
        A[2,:] .= β
        yn = A*yn
        yn[2] += σp*randn()
        y[i] = yn[2]
    end
    
    # Solution can be non-stationary, appears to grow exponentially
    # very quick test to reject un-predictable series
    if abs((mean(y[N÷2+1:end]) - mean(y[1:N÷2]))/std(y[1:N÷2])) < 3
        return mean(y), std(y)
    else
        return NaN, NaN
    end
end


# ============================================================
# ===================   Hyper Parameters   ===================
# ============================================================

coins = ["AAVE", "AVAX", "BAT", "BCH", "BTC", "CRV", "DOGE", "ETH", "GRT", "LINK", "LTC", "MKR", "SUSHI", "TRUMP", "UNI", "XRP", "XTZ", "PEPE", "SHIB"] # , "DOT", "SOL", "YFI"

@load "liveVersion/bot1Paper/Account.jld2" Account parms

# Variables
# θi = [5, 6, 28, 20, 17, 20, 5, 0.453471586]
# θi = [5, 6, 28, 10, 150, 12, 7, 1.2567906333910892]
# θi = [5, 6, 28, 48, 38, 3, 5, 1.357056078482785]
θi = [5, 6, 28, 10, 155, 8, 5, 2.1]
parms = Dict(
    "tStep"       => Minute(θi[1]),
    "step"        => Hour(θi[2]),
    "trainPeriod" => Day(θi[3]),
    "testPeriod"  => Day(θi[4]),
    "MW"          => Hour(θi[5]),
    "P"           => Int(θi[6]),
    "MTS"         => Int(θi[7]),
    "portion"     => 1,
    # Increased fees to push it harder
    "TakerFee"    => 0.0025+eps(),
    "MakerFee"    => 0.0015+eps(),
)
parms["portion"] = parms["step"]/(parms["MTS"]*parms["MW"])*θi[8]

# ============================================================
# =================   Compute Predictions   ==================
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
    transform!(data, :t => ByRow(t -> round(t, parms["tStep"], RoundDown)) => :t)
    minData = combine(groupby(data, :t),
        :bp => last => :bp,
    )
    minData = leftjoin(DataFrame(t = range(minData.t[1],minData.t[end],step=parms["tStep"])), minData, on = :t)
    sort!(minData, :t)
    while any(ismissing.(minData.bp))
        idx = findall(ismissing, minData.bp)
        minData.bp[idx] = minData.bp[idx .- 1]
    end
    transform!(minData, :bp => ByRow(log) => :X)

    predictions = Dict[]
    T0 = round(minData.t[1], Day, RoundDown)+parms["trainPeriod"]+parms["step"]
    MaxT = minData.t[end]-parms["testPeriod"]

    for (i, T) in enumerate(range(T0, MaxT, step=parms["step"])) |> ProgressBar
        TS = @view minData[T-parms["trainPeriod"] .≤ minData.t .< T,:]
        μ = mean(TS.X)
        σ = std(TS.X)
        X = (TS.X .- μ) ./ σ

        x,y = Format_TS_data(X, parms["P"], 0)
        μ0, V0 = arPost(x,y)
        x,y = Format_TS_data(X, parms["P"], 1)
        μ1, V1 = arPost(x,y)
        x,y = Format_TS_data(X, parms["P"], 2)
        μ2, V2 = arPost(x,y)

        # If any predictions fail, skip
        if all(isfinite.([μ0, V0, μ1, V1, μ2, V2]))
            push!(predictions,
                Dict(
                    "t" => T,
                    "μ0" => μ0*σ + μ,
                    "V0" => V0*σ,
                    "μ1" => μ1*σ,
                    "V1" => V1*σ,
                    "μ2" => μ2*σ,
                    "V2" => V2*σ,
                    "c"  => TS.X[end]
                )
            )
        end
    end

    preds[coin] = DataFrame(predictions)
end

@save "bot/data/bot1.jld2" preds

# coin = coins[3]
# [count(isnan.(Matrix(select(preds[c], Not(:t))))) for c in coins] |> sum

# ============================================================
# ============   Create predictor for TP and BP   ============
# ============================================================

# Get 'ideal' trade outcomes and try use preds to predict them

@load "bot/data/bot1.jld2" preds
@load "bot/data/tunBot_2_qutBar.jld2" preds

# To avoid over-fitting we limit this training step to a smaller period of time.
T0 = minimum([preds[c].t[1] for c in coins]) # Find first valid prediction time in data
MaxT = T0 + Year(1) # Time to stop
# T0 = minimum([preds[c].t[1] for c in coins]) + Year(1) + Month(2) # Find first valid prediction time in data
# MaxT = maximum([preds[c].t[end] for c in coins]) # Time to stop
X = Matrix{Float64}(undef, 0, 7)
Y = Matrix{Float64}(undef, 0, 2)

for coin in coins
    println("Simulating $coin")

    # Get real coin value movement from bars
    barData = loadCryptoBars(coin)
    bestBuys = Dict[]
    times = preds[coin][preds[coin].t .< MaxT,:].t

    # ensure there is predictions in specified period. New coins is the issue.
    if isempty(times)
        continue
    end

    y = Matrix{Float64}(undef, length(times), 2)

    for (i, T) in enumerate(times) |> ProgressBar
        # Get training subset
        TS = @view barData[T .< barData.t .≤ T+parms["testPeriod"], :]

        # 20% lowest price in proceeding period MW
        # BBP = quantile(TS.h[TS.t .≤ T+parms["MW"]], 0.2)
        # When does BBP first occur
        # BBT = findfirst(TS.h .≤ BBP)
        # After BBP 80% highest price achieved
        # BTP = quantile(TS.l[BBT:end], 0.8)
        # 20% lowest price in proceeding period MW
        BBP = quantile(TS.l[TS.t .≤ T+parms["MW"]], 0.2)
        # When does BBP first occur
        BBT = findfirst(TS.l .≤ BBP)
        # After BBP 80% highest price achieved
        BTP = quantile(TS.h[BBT:end], 0.8)

        # Alternate idea, performed worse
        # BBP, BBT = findmin(TS2.l[TS2.t .≤ T+MW])
        # BTP = maximum(TS2.h[BBT:end])

        y[i,:] .= log.([BTP, BBP])
    end

    X = [X ; Matrix(preds[coin][preds[coin].t .< MaxT,[:μ0,:V0,:μ1,:V1,:μ2,:V2,:c]])]
    Y = [Y ; y]
end

# Make X a design matrix
X = [ones(size(X,1)) X]
# OLS solution f : X → Y
β = (X'*X+0I(8)) \ X'*Y

@save "bot/data/bot_weights.jld2" β
@save "bot/data/bot_weights2.jld2" β

ŷ = X*β
Y - ŷ
count(ŷ[:,1] .< ŷ[:,2])
count(Y[:,1] .< Y[:,2])

X1 = X[:,1:end-1]
β1 = (X1'*X1) \ X1'*Y
Y - X1*β1

# ============================================================
# ================   Compute Trade Outcomes   ================
# ============================================================

@load "bot/data/bot_weights.jld2" β
@load "bot/data/bot_weights2.jld2" β
@load "bot/data/bot1.jld2" preds
trades = Dict{String, DataFrame}()

for coin in coins
    println("Simulating $coin")
    barData = loadCryptoBars(coin)
    buyOutcomes = Dict[]

    for T in preds[coin].t |> ProgressBar
        # Estimate take profit and buy price
        TP, BP = hcat(1, Matrix(select(preds[coin][preds[coin].t .== T, :], [:μ0,:V0,:μ1,:V1,:μ2,:V2,:c])))*β

        # Get bar data to determine outcome
        TS2 = @view barData[T .< barData.t .≤ T+parms["testPeriod"], :]

        # Find when/if buy executes
        BT = findfirst(TS2.l[TS2.t .≤ T+parms["MW"]] .< exp(BP))
        # BT = findfirst(TS2.h[TS2.t .≤ T+parms["MW"]] .< exp(BP))
        if BT !== nothing # If buy executes
            # Check if take profit is hit
            WT = findfirst(TS2.h[BT:end] .> exp(TP))
            # WT = findfirst(TS2.l[BT:end] .> exp(TP))
            if WT !== nothing # If take profit is hit
                # open price < hopeful buy price => market order, enter postion at open price
                FP = TS2.o[1] < exp(BP) ? TS2.o[1] : exp(BP)
                push!(buyOutcomes, 
                    Dict(
                        "startTime"     => T,
                        "coin"          => coin,
                        "buyPrice"      => exp(BP),
                        "takeProfit"    => exp(TP),
                        "startPrice"    => TS2.o[1],
                        "buyTime"       => TS2.t[BT],
                        "avgFillPrice"  => FP,
                        "orderType"     => TS2.o[1] < exp(BP) ? "market" : "limit",
                        "status"        => "trade completed",
                        "avgSellPrice"  => exp(TP),
                        "completeTime"  => TS2.t[BT+WT-1],
                        "PL_pc"         => exp(TP)/FP,
                        "value"         => 0.0,
                        "orderQty"      => 0.0,
                        "sellQty"       => 0.0,
                        "totalFees"     => 0.0,
                        "buyFee"        => 0.0,
                        "sellFee"       => 0.0,
                        "PL_dollars"    => 0.0,
                    )
                )
            else # Take profit is not hit => kill at T + testPeriod
                FP = TS2.o[1] < exp(BP) ? TS2.o[1] : exp(BP)
                push!(buyOutcomes, 
                    Dict(
                        "startTime"     => T,
                        "coin"          => coin,
                        "buyPrice"      => exp(BP),
                        "takeProfit"    => exp(TP),
                        "startPrice"    => TS2.o[1],
                        "buyTime"       => TS2.t[BT],
                        "avgFillPrice"  => FP,
                        "orderType"     => TS2.o[1] < exp(BP) ? "market" : "limit",
                        "status"        => "trade killed",
                        "avgSellPrice"  => TS2.c[end],
                        "completeTime"  => TS2.t[end],
                        "PL_pc"         => TS2.c[end]/FP,
                        "value"         => 0.0,
                        "orderQty"      => 0.0,
                        "sellQty"       => 0.0,
                        "totalFees"     => 0.0,
                        "buyFee"        => 0.0,
                        "sellFee"       => 0.0,
                        "PL_dollars"    => 0.0,
                    )
                )
            end
        else # Buy fails to execute
            push!(buyOutcomes, 
                Dict(
                    "startTime"     => T,
                    "coin"          => coin,
                    "buyPrice"      => exp(BP),
                    "takeProfit"    => exp(TP),
                    "startPrice"    => TS2.o[1],
                    "buyTime"       => DateTime(0),
                    "avgFillPrice"  => 0.0,
                    "orderType"     => "limit",
                    "status"        => "order failed",
                    "avgSellPrice"  => 0.0,
                    "completeTime"  => T+parms["MW"],
                    "PL_pc"         => 1.0,
                    "value"         => 0.0,
                    "orderQty"      => 0.0,
                    "sellQty"       => 0.0,
                    "totalFees"     => 0.0,
                    "buyFee"        => 0.0,
                    "sellFee"       => 0.0,
                    "PL_dollars"    => 0.0,
                )
            )
        end
    end

    trades[coin] = DataFrame(buyOutcomes)
end

@save "bot/data/bot1.jld2" preds trades

# ============================================================
# ==================   Back-test Strategy   ==================
# ============================================================

@load "bot/data/bot1.jld2" preds trades

parms["MakerFee"] = 0.0015
parms["TakerFee"] = 0.0025

b0 = 0.8123960408246376
b1 = 0.20159852269506473

T0 = minimum([trades[c].startTime[1] for c in  keys(trades)])
MaxT = maximum([trades[c].startTime[end] for c in keys(trades)])
n = length(range(T0, MaxT, step=parms["step"]))

begin
    liquidity = 2_000
    V = vcat(liquidity, zeros(n)) # log
    Liq = vcat(liquidity, zeros(n)) # log

    ActiveTrades = Dict[]
    Trades = 0
    failures = 0

    for (i, T) in enumerate(range(T0, MaxT, step=parms["step"]))
        # Complete trades which have finsihed before T
        V[i+1] = copy(V[i])
        for j in findall(x -> x["completeTime"] <= T, ActiveTrades)
            X = ActiveTrades[j]
            Trades += 1
            if X["status"] == "order failed"
                # order failed to execute, add value back to liquidity
                failures += 1
                liquidity += X["Value"]
            else
                # trade completed, add profit to value, add value to liquidity
                fee = (X["orderType"] == "limit" ? 1-parms["MakerFee"] : 1-parms["TakerFee"])*(1-parms["MakerFee"])
                V[i+1] += (X["PL_pc"] - 1)*fee*X["Value"]
                liquidity += X["PL_pc"]*fee*X["Value"]
            end
        end
        filter!(x -> x["completeTime"] > T, ActiveTrades)

        # Get best trading opportunities
        currentTrades = [trades[c][trades[c].startTime .== T,:] for c in keys(trades)]
        filter!(x -> .! isempty(x), currentTrades)
        filter!(x -> x.takeProfit[1]/x.buyPrice[1] > 1.05, currentTrades)
        
        tv = [isempty(t) ? 0 : t.takeProfit[1]/t.buyPrice[1] for t in currentTrades]
        # tv = [isempty(t) ? 0 : t.startPrice[1]/t.buyPrice[1] for t in currentTrades]
        # tv = [isempty(t) ? 0 : b0 + b1 * t.takeProfit[1]/t.buyPrice[1] for t in currentTrades]

        idx = sortperm(tv, rev = true)
        # idx = sortperm(tv)
        # activate trade 
        for t in currentTrades[idx[1:min(parms["MTS"], length(tv))]]
            if liquidity > 1
                push!(ActiveTrades, 
                    Dict(
                        "Value" => min(V[i+1]*parms["portion"], liquidity),
                        "completeTime" => t.completeTime[1],
                        "PL_pc" => t.PL_pc[1],
                        "status" => t.status[1],
                        "orderType" => t.orderType[1]
                    )
                )
                liquidity -= min(V[i+1]*parms["portion"], liquidity)
            end
        end

        Liq[i+1] = liquidity
    end
end

plot(V, legend=false)
plot!(Liq)
plot(Liq./V)

Vend = copy(V[end])
for X in ActiveTrades
    # Trades += 1
    if X["status"] != "order failed"
        # trade completed, add profit to value, add value to liquidity
        fee = (X["orderType"] == "limit" ? 1-parms["MakerFee"] : 1-parms["TakerFee"])*(1-parms["MakerFee"])
        Vend += (X["PL_pc"] - 1)*fee*X["Value"]
    end
end
Vend

# Estimate Sharpe ratio
rfr = 1.0436                    # risk-free return
returns = exp.(diff(log.(V)))   # return after each step
rop = mean(returns)             # mean return each step
σp = std(returns)               # std of return each step
σs = std(returns[returns .< 1]) # std of negative returns
p = 24*365/parms["step"].value  # trades performed in 1 year
ASR = √p*(rop - rfr.^(1/p))/σp  # Annualised Sharpe ratio
ASoR = √p*(rop - rfr.^(1/p))/σs # Annualised Sortino ratio

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


