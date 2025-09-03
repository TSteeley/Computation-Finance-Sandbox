#=
    bot1_1.jl

    Author: Thomas P. Steele
=#

using Distributions, StatsBase, ProgressBars
using FinData, Dates, DataFrames
using LinearAlgebra, Base.Threads, JLD2
using Plots, CustomPlots
include("../functions.jl")

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
        A[2,:] .= mβ + σβ*randn(m)*σ
        yn = A*yn
        yn[2] += σ*randn()
    end

    for i in 1:N
        σ = 1/sqrt(rand(pΛ))
        A[2,:] .= mβ + σβ*randn(m)*σ
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

# ============================================================
# ===================   Hyper Parameters   ===================
# ============================================================

coins = ["AVAX", "BAT", "BCH", "BTC", "CRV", "DOGE", "ETH", "GRT", "LINK", "LTC", "MKR", "SUSHI", "TRUMP", "UNI", "XRP", "XTZ", "PEPE", "SHIB", "YFI", "DOT"] # , "SOL", "AAVE"

@load "liveVersion/bot1Paper/Account.jld2" parms

# Variables
# θi = [5, 6, 28, 20, 17, 20, 5, 0.453471586]
# θi = [5, 6, 28, 10, 150, 12, 7, 1.2567906333910892]
# θi = [5, 6, 28, 48, 38, 3, 5, 1.357056078482785]
# θi = [5, 6, 28, 10, 155, 8, 5, 2.1]
θi = [5, 6, 28, 10, 32, 14, 5, 2.53]
θi = [5, 6, 28, 14, 72, 5, 5, 0.53]
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
# ================   Load and process data   =================
# ============================================================

quoteData = Dict{String, DataFrame}()
seriesData = Dict{String, DataFrame}()
for c in coins |> ProgressBar
    data = loadCryptoQuotes(c)
    data = data[data.ap .!= 0,:]
    
    # First is data to work on
    data[!,:time] = copy(data.t)
    data[!,:t] = round.(data.time, Minute(1), RoundDown)
    data = combine(groupby(data, :t),
        :ap => minimum => :ap,
        :bp => maximum => :bp,
        [:time,:as,:bs] => twapBias(Minute(1)) => :bias,
        [:time,:as,:ap,:bs,:bp] => twapPrice(Minute(1)) => :vwap,
        :time => length => :n,
        :as => sum => :as,
        :bs => sum => :bs,
    )
    quoteData[c] = data
    
    # data to infer on
    TS = copy(data[!,["t", "vwap"]])
    TS.t = round.(TS.t, parms["tStep"], RoundDown)
    TS = combine(groupby(TS, :t), :vwap => last => :vwap)
    TS[!,"vwap"] = log.(TS[!,"vwap"])
    TS = leftjoin(DataFrame(t = range(TS.t[1],TS.t[end],step=parms["tStep"])), TS, on = :t)
    
    TS = TS[sortperm(TS.t),:]
    while any(ismissing.(TS.vwap))
        idx = findall(ismissing, TS.vwap)
        TS.vwap[idx] = TS.vwap[idx .- 1]
    end
    seriesData[c] = TS
end

# data[608101,:]

# Data = combine(groupby(data, :t),
#     :ap => minimum => :ap,
#     :bp => maximum => :bp,
#     [:time,:as,:bs] => twapBias(Minute(1)) => :bias,
#     [:time,:as,:ap,:bs,:bp] => twapPrice(Minute(1)) => :vwap,
#     :time => length => :n,
#     :as => sum => :as,
#     :bs => sum => :bs,
# )

# findfirst(isnan.(Data.vwap))

# data.time[5]

# mean(isfinite.(quoteData["AVAX"].vwap))
# findfirst(isnan.(quoteData["AVAX"].vwap))

# groupby(data, :t)[608101].time[1]

# ============================================================
# =================   Compute Predictions   ==================
# ============================================================

# Store predictions; dictionary of DataFrames
# Either start from scratch or continue from already done predictions
@load "bot/data/bot1_1.jld2" preds
preds = Dict{String, DataFrame}()

for coin in coins
    if coin ∈ keys(preds)
        println("Skip $coin")
        continue
    end
    println("$coin")

    predictions = Dict[]
    T0 = round(quoteData[coin].t[1], Day, RoundDown)+parms["trainPeriod"]+parms["step"]
    MaxT = quoteData[coin].t[end]-parms["testPeriod"]

    for (i, T) in enumerate(range(T0, MaxT, step=parms["step"])) |> ProgressBar
        TS = @view seriesData[coin][T-parms["trainPeriod"] .≤ seriesData[coin].t .< T,:]
        μ = mean(TS.vwap)
        σ = std(TS.vwap)
        σ != 0 || continue
        X = (TS.vwap .- μ) ./ σ

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
                    "c"  => TS.vwap[end]
                )
            )
        end
    end

    preds[coin] = DataFrame(predictions)
end

@save "bot/data/bot1_1.jld2" preds

# mean(isfinite.(seriesData["AVAX"].vwap))



# coin = coins[3]
# [count(isnan.(Matrix(select(preds[c], Not(:t))))) for c in coins] |> sum

# ============================================================
# ============   Create predictor for TP and BP   ============
# ============================================================

# Get 'ideal' trade outcomes and try use preds to predict them

@load "bot/data/bot1_1.jld2" preds
# @load "bot/data/tunBot_2_qutBar.jld2" preds

# To avoid over-fitting we limit this training step to a smaller period of time.
T0 = minimum([preds[c].t[1] for c in coins]) # Find first valid prediction time in data
# T0 = minimum([preds[c].t[1] for c in coins]) + Year(1) + Month(2) # Find first valid prediction time in data
MaxT = T0 + Year(1) # Time to stop
# MaxT = maximum([preds[c].t[end] for c in coins]) # Time to stop
X = Matrix{Float64}(undef, 0, 7)
Y = Matrix{Float64}(undef, 0, 2)

for coin in coins
    println("Simulating $coin")

    bestBuys = Dict[]
    times = preds[coin][T0 .< preds[coin].t .< MaxT,:].t

    # ensure there is predictions in specified period. New coins is the issue.
    if isempty(times)
        continue
    end

    y = Matrix{Float64}(undef, length(times), 2)
    # x = Matrix{Float64}(undef, length(times), 7)
    incl = Int[]

    for (i, T) in enumerate(times) |> ProgressBar
        # Get training subset
        TS = @view quoteData[coin][T .< quoteData[coin].t .≤ T+parms["testPeriod"], :]
        .! isempty(TS) || continue
        push!(incl, i)

        # 20% lowest price in proceeding period MW
        # BBP = quantile(TS.h[TS.t .≤ T+parms["MW"]], 0.2)
        # When does BBP first occur
        # BBT = findfirst(TS.h .≤ BBP)
        # After BBP 80% highest price achieved
        # BTP = quantile(TS.l[BBT:end], 0.8)
        # 20% lowest price in proceeding period MW
        BBP = quantile(TS.ap[TS.t .≤ T+parms["MW"]], 0.2)
        # When does BBP first occur
        BBT = findfirst(TS.ap .≤ BBP)
        # After BBP 80% highest price achieved
        BTP = quantile(TS.bp[BBT:end], 0.8)

        # Alternate idea, performed worse
        # BBP, BBT = findmin(TS2.l[TS2.t .≤ T+MW])
        # BTP = maximum(TS2.h[BBT:end])

        Y = [Y ; log.([BTP, BBP])']
        # y[i,:] .= log.([BTP, BBP])
        # x = preds[]
    end

    X = [X ; Matrix(preds[coin][T0 .< preds[coin].t .< MaxT,[:μ0,:V0,:μ1,:V1,:μ2,:V2,:c]])[incl,:]]
    # Y = [Y ; y]
end

# Make X a design matrix
X = [ones(size(X,1)) X]
# OLS solution f : X → Y
β = (X'*X+0I(8)) \ X'*Y

# @save "bot/data/bot_weights.jld2" β
# @save "bot/data/bot_weights2.jld2" β

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
# @load "bot/data/bot_weights2.jld2" β
@load "bot/data/bot1_1.jld2" preds
trades = Dict{String, DataFrame}()

for coin in coins
    println("Simulating $coin")
    buyOutcomes = Dict[]

    for T in preds[coin].t |> ProgressBar
        # Get bar data to determine outcome
        TS = @view quoteData[coin][T .< quoteData[coin].t .≤ T+parms["testPeriod"], :]
        .! isempty(TS) || continue
        # Estimate take profit and buy price
        pred = preds[coin][preds[coin].t .== T, :]
        TP, BP = hcat(1, Matrix(select(pred, [:μ0,:V0,:μ1,:V1,:μ2,:V2,:c])))*β

        # TS2 = @view barData[T .< barData.t .≤ T+parms["testPeriod"], :]

        # Find when/if buy executes
        BT = findfirst(TS.ap[TS.t .≤ T+parms["MW"]] .< exp(BP))
        if BT !== nothing # If buy executes
            # Check if take profit is hit
            WT = findfirst(TS.bp[BT:end] .> exp(TP))
            if WT !== nothing # If take profit is hit
                # open price < hopeful buy price => market order, enter postion at open price
                o = quoteData[coin][T .< quoteData[coin].t,"ap"][1]
                FP = o < exp(BP) ? o : exp(BP)
                push!(buyOutcomes, 
                    Dict(
                        "startTime"     => T,
                        "coin"          => coin,
                        "buyPrice"      => exp(BP),
                        "takeProfit"    => exp(TP),
                        "startPrice"    => o,
                        "buyTime"       => TS.t[BT],
                        "avgFillPrice"  => exp(BP),
                        "orderType"     => o < exp(BP) ? "market" : "limit",
                        "status"        => "trade completed",
                        "avgSellPrice"  => exp(TP),
                        "completeTime"  => TS.t[BT+WT-1],
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
                o = quoteData[coin][T .< quoteData[coin].t,"ap"][1]
                c = quoteData[coin][quoteData[coin].t .< T + parms["testPeriod"],"bp"][end]
                FP = o < exp(BP) ? o : exp(BP)
                push!(buyOutcomes, 
                    Dict(
                        "startTime"     => T,
                        "coin"          => coin,
                        "buyPrice"      => exp(BP),
                        "takeProfit"    => exp(TP),
                        "startPrice"    => o,
                        "buyTime"       => TS.t[BT],
                        "avgFillPrice"  => FP,
                        "orderType"     => "limit",
                        "status"        => "trade killed",
                        "avgSellPrice"  => c,
                        "completeTime"  => TS.t[end],
                        "PL_pc"         => c/FP,
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
            o = quoteData[coin][T .< quoteData[coin].t,"ap"][1]
            push!(buyOutcomes, 
                Dict(
                    "startTime"     => T,
                    "coin"          => coin,
                    "buyPrice"      => exp(BP),
                    "takeProfit"    => exp(TP),
                    "startPrice"    => o,
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

@save "bot/data/bot1_1.jld2" preds trades

# ============================================================
# ===================   Classify winners   ===================
# ============================================================
@load "bot/data/bot1_1.jld2" preds trades

X = Matrix{Float64}(undef, 0, 9)
Y = Float64[]

for c in coins
    A = leftjoin(preds[c], rename(trades[c], "startTime" => "t"), on="t")
    dropmissing!(A)
    X = [X ; Matrix(select(A, ["μ0", "V0", "μ1", "V1", "μ2", "V2", "c", "buyPrice", "takeProfit"]))]
    Y = append!(Y, trades[c][!,"PL_pc"] .> 1.05)
end

Y2 = clamp.(Y, eps(), 1-eps())
X = [ones(size(X,1)) X]
X[:,end-1:end] = log.(X[:,end-1:end])

sigmoid(x) = 1/(1+exp(-x))
softmax(x) = x > 0.5 ? 1 : 0

using CUDA

lossFun(ŷ::Vector) = -Y2'*log.(ŷ) - (1 .- Y2)'*log.(1 .- ŷ)
lossFun(ŷ::CuArray) = -yt2'*log.(ŷ) - (1 .- yt2)'*log.(1 .- ŷ)
sigmoid(X::CuArray) = 1 ./ (1 .+ exp.(-X))
# n = size(X, 1)

xt = CuArray{Float32}(X)
yt = CuArray{Float32}(Y)
yt2 = CuArray{Float32}(clamp.(yt, eps(1.0f0), 1.0f0-eps(1.0f0)))

γ = CUDA.randn(size(xt,2))
γ = CuArray{Float32}(γ)
ŷ = sigmoid.(xt*γ)
losses = Float64[lossFun(clamp.(ŷ, eps(1.0f0), 1.0f0-eps(1.0f0)))]

B = xt'*xt

for _ in 1:100_000 |> ProgressBar
    γ -= (ŷ'*(1 .- ŷ)*B)\xt'*(ŷ - yt)
    ŷ = sigmoid(xt*γ)
    push!(losses, lossFun(clamp.(ŷ, eps(1.0f0), 1-eps(1.0f0))))
end

plot(losses)

γ = Vector{Float64}(γ)
ŷ = softmax.(sigmoid.(X*γ))
Y2 = softmax.(Y)

mean((ŷ .== 1) .& (Y2 .== 1))
mean((ŷ .== 0) .& (Y2 .== 0))
mean((ŷ .== 1) .& (Y2 .== 0))
mean((ŷ .== 0) .& (Y2 .== 1))

@save "bot/data/bot1_infWeight.jld2" γ

# ============================================================
# ==================   Back-test Strategy   ==================
# ============================================================

@load "bot/data/bot1_1.jld2" preds trades
@load "bot/data/bot1_infWeight.jld2" γ

sigmoid(x) = 1/(1+exp(-x))

parms["MakerFee"] = 0.0015
parms["TakerFee"] = 0.0025

b0 = 0.8123960408246376
b1 = 0.20159852269506473

T0 = minimum([trades[c].startTime[1] for c in keys(trades)])
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
        # currentTrades = []
        # tv = Float64[]
        # for c in coins
        #     t = trades[c][trades[c].startTime .== T,:]
        #     p = preds[c][preds[c].t .== T,:]
        #     if .! any(isempty.([t,p]))
        #         push!(currentTrades, t)
        #         push!(tv,sigmoid([1 ; Vector(p[1,["μ0", "V0", "μ1", "V1", "μ2", "V2", "c"]]); log.(Vector(t[1,["buyPrice", "takeProfit"]]))]'*γ))
        #     end
        # end
        
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
        # i+=1
        # T=range(T0, MaxT, step=parms["step"])[i]
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

@load "bot/data/bot1_1.jld2" preds trades

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


