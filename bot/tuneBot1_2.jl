#=
    tuneBot1_2.jl

    Trying a more direct Markov chain approach

    Author: Thomas P. Steele
=#

using Distributions, StatsBase, ProgressBars
using FinData, Dates, DataFrames
using LinearAlgebra, Base.Threads, JLD2
using Plots, CustomPlots, Random
include("../functions.jl")

coins = ["AAVE", "AVAX", "BAT", "BCH", "BTC", "CRV", "DOGE", "ETH", "GRT", "LINK", "LTC", "MKR", "SUSHI", "TRUMP", "UNI", "XRP", "XTZ"] # , "DOT","SOL", "YFI", "PEPE", "SHIB"

# ============================================================
# ===================   Helper Functions   ===================
# ============================================================

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

function arPost(X::AbstractMatrix{<:Real}, Y::AbstractVector{<:Real}; N::Int=50_000)
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

# ============================================================
# =====================   Load in Data   =====================
# ============================================================
# Loading in the data takes about 10 seconds per coin
# Pre-load the data to use in each iteration
# saves up to 190 seconds per iteration at a pretty good price

function loadData(tStep::Minute)
    quoteData = Dict{String, DataFrame}()
    barData = Dict{String, DataFrame}()
    for coin in coins |> ProgressBar
        data = loadCryptoQuotes(coin)
        filter!(d -> d.bp != 0, data)
        transform!(data, :t => ByRow(t -> round(t, tStep, RoundDown)) => :t)
        data = combine(groupby(data, :t),
            :bp => last => :bp,
        )
        data = leftjoin(DataFrame(t = range(data.t[1],data.t[end],step=tStep)), data, on = :t)
        sort!(data, :t)
        while any(ismissing.(data.bp))
            idx = findall(ismissing, data.bp)
            data.bp[idx] = data.bp[idx .- 1]
        end
        transform!(data, :bp => ByRow(log) => :X)
        select!(data, Not(:bp))
        quoteData[coin] = data

        barDatum = loadCryptoBars(coin)
        select!(barDatum, Not([:v, :vw, :n]))
        barData[coin] = barDatum
    end
    return quoteData, barData
end

# sizeof(barData.c)
# sizeof(minData.t)

# 19 coins
# X 1_907_432 bytes
# t 1_907_432 bytes
# Quotes up to abt 72MB

# 1 col of barData 8_745_128 byts
# Bar data up to abt 831MB

# abt 900 MB of RAM for all data, should be fine.

# ============================================================
# =================   Compute Predictions   ==================
# ============================================================

function predict(parms::Dict, quoteData::Dict)
    # Store predictions; dictionary of DataFrames
    # Either start from scratch or continue from already done predictions
    preds = Dict{String, DataFrame}()
    
    for coin in coins
        T0 = round(quoteData[coin].t[1], Day, RoundDown)+parms["trainPeriod"]+parms["step"]
        MaxT = quoteData[coin].t[end]-parms["testPeriod"]
        predictions = Dict[]
    
        for T in range(T0, MaxT, step=parms["step"])
        # for (i, T) in enumerate(range(T0, MaxT, step=parms["step"]))
            # println(i)
            TS = @view quoteData[coin][T-parms["trainPeriod"] .≤ quoteData[coin].t .< T,:]
            μ = mean(TS.X)
            σ = std(TS.X)
            σ != 0 || continue
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
    return preds
end

# ============================================================
# ============   Create predictor for TP and BP   ============
# ============================================================

# Get 'ideal' trade outcomes and try use preds to predict them
function createPredictor(parms::Dict, preds::Dict, barData::Dict)
    # To avoid over-fitting we limit this training step to a smaller period of time.
    T0 = minimum([preds[c].t[1] for c in coins]) # Find first valid prediction time in data
    MaxT = T0 + Year(1) # Time to stop
    # T0 = minimum([preds[c].t[1] for c in coins]) + Year(1) + Month(2) # Find first valid prediction time in data
    # MaxT = maximum([preds[c].t[end] for c in coins]) # Time to stop
    X = Matrix{Float64}(undef, 0, 7)
    Y = Matrix{Float64}(undef, 0, 2)
    
    for coin in coins
        # Get real coin value movement from bars
        times = preds[coin][preds[coin].t .< MaxT,:].t
    
        # ensure there is predictions in specified period. New coins is the issue.
        if isempty(times)
            continue
        end
    
        y = Matrix{Float64}(undef, length(times), 2)
    
        for (i, T) in enumerate(times)
            # Get training subset
            TS = @view barData[coin][T .< barData[coin].t .≤ T+parms["testPeriod"], :]
    
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
    return (X'*X) \ X'*Y
end

# ============================================================
# ================   Compute Trade Outcomes   ================
# ============================================================

function getTradeOutcomes(parms::Dict, preds::Dict, barData::Dict, β::Matrix)
    trades = Dict{String, DataFrame}()
    
    for coin in coins
        buyOutcomes = Dict[]
    
        for T in preds[coin].t
            # Estimate take profit and buy price
            TP, BP = hcat(1, Matrix(select(preds[coin][preds[coin].t .== T, :], [:μ0,:V0,:μ1,:V1,:μ2,:V2,:c])))*β
    
            # Get bar data to determine outcome
            TS2 = @view barData[coin][T .< barData[coin].t .≤ T+parms["testPeriod"], :]
    
            # Find when/if buy executes
            BT = findfirst(TS2.l[TS2.t .≤ T+parms["MW"]] .< exp(BP))
            if BT !== nothing # If buy executes
                # Check if take profit is hit
                WT = findfirst(TS2.h[BT:end] .> exp(TP))
                if WT !== nothing # If take profit is hit
                    # open price < hopeful buy price => market order, enter postion at open price
                    FP = TS2.o[1] < exp(BP) ? TS2.o[1] : exp(BP)
                    push!(buyOutcomes, 
                        Dict(
                            "startTime"     => T,
                            "buyPrice"      => exp(BP),
                            "takeProfit"    => exp(TP),
                            "orderType"     => TS2.o[1] < exp(BP) ? "market" : "limit",
                            "status"        => "trade completed",
                            "completeTime"  => TS2.t[BT+WT-1],
                            "PL_pc"         => exp(TP)/FP,
                        )
                    )
                else # Take profit is not hit => kill at T + testPeriod
                    FP = TS2.o[1] < exp(BP) ? TS2.o[1] : exp(BP)
                    push!(buyOutcomes, 
                        Dict(
                            "startTime"     => T,
                            "buyPrice"      => exp(BP),
                            "takeProfit"    => exp(TP),
                            "orderType"     => TS2.o[1] < exp(BP) ? "market" : "limit",
                            "status"        => "trade killed",
                            "completeTime"  => TS2.t[end],
                            "PL_pc"         => TS2.c[end]/FP,
                        )
                    )
                end
            else # Buy fails to execute
                push!(buyOutcomes, 
                    Dict(
                        "startTime"     => T,
                        "buyPrice"      => exp(BP),
                        "takeProfit"    => exp(TP),
                        "orderType"     => "limit",
                        "status"        => "order failed",
                        "completeTime"  => T+parms["MW"],
                        "PL_pc"         => 1.0,
                    )
                )
            end
        end
    
        trades[coin] = DataFrame(buyOutcomes)
    end
    return trades
end

# ============================================================
# ==================   Back-test Strategy   ==================
# ============================================================

function runBackTest(parms::Dict, trades::Dict)
    T0 = minimum([trades[c].startTime[1] for c in keys(trades)])
    MaxT = maximum([trades[c].startTime[end] for c in keys(trades)])
    n = length(range(T0, MaxT, step=parms["step"]))

    liquidity = 2_000
    V = [liquidity ; zeros(n)]
    ActiveTrades = Dict[]
    trads = 0
    fails = 0

    for (i, T) in enumerate(range(T0, MaxT, step=parms["step"]))
        V[i+1] = V[i]
        # Complete trades which have finsihed before T
        for j in findall(x -> x["completeTime"] <= T, ActiveTrades)
            X = ActiveTrades[j]
            # Trades += 1
            if X["status"] == "order failed"
                # order failed to execute, add value back to liquidity
                fails += 1
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

        idx = sortperm(tv, rev = true)
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
                trads += 1
            end
        end
    end

    # Finish up still active trades to get final value of trading alg
    Vend = copy(V[end])
    for X in ActiveTrades
        # Trades += 1
        if X["status"] != "order failed"
            # trade completed, add profit to value, add value to liquidity
            fee = (X["orderType"] == "limit" ? 1-parms["MakerFee"] : 1-parms["TakerFee"])*(1-parms["MakerFee"])
            Vend += (X["PL_pc"] - 1)*fee*X["Value"]
        end
    end

    # Estimate Sharpe ratio
    rfr = 1.0436 # risk-free return
    returns = exp.(diff(log.(V)))   # return after each step
    rop = mean(returns)             # mean return each step
    σp = std(returns)               # std of return each step
    σs = std(returns[returns .< 1]) # std of negative returns
    p = 24*365/parms["step"].value  # trades performed in 1 year
    ASR = √p*(rop - rfr.^(1/p))/σp  # Annualised Sharpe ratio
    ASoR = √p*(rop - rfr.^(1/p))/σs # Annualised Sortino ratio

    # Return
    # - Final portfolio value
    # - Annualised Sharpe ratio
    # - Percentage of trades which were rejected
    return Vend, ASR, ASoR, fails/trads
end

# ============================================================
# ========================   Main   ==========================
# ============================================================
# Because loading up the data takes soo long, we are aiming to 
# do that the least odd decisions are probably because of this.

# I am using my own blend of MCMC, and SMC to try to optimise 
# the parameters. These aren't optimisation algorithms, but I
# think the way they search the parameter space is what I want.

priors = [
    DiscreteUniform(1, 180), # tStep:: Binning of quote data in minutes
    DiscreteUniform(1, 168), # step:: How often trades are performed in hours
    DiscreteUniform(1, 60),  # trainPeriod:: how much historical data to train on in days
    DiscreteUniform(1, 60),  # testPeriod:: How long a trade can live for before being killed in days
    DiscreteUniform(1, 168), # MW:: Max waiting time for a limit order in hours
    DiscreteUniform(1, 20),  # P:: Parameter for AR model
    DiscreteUniform(1, 19),  # MTS:: max trades started at once
    Exponential(1),          # portion:: What portion of portfolio value is put down on each trade
]

prior = product_distribution(priors...)

log_p(θ::Vector) = log_p(prior, θ)

# q()
# log_q()

function model(θi::Vector, quoteData::Dict, barData::Dict)
    # algorithm parameters
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
        "TakerFee"    => 0.0055,
        "MakerFee"    => 0.0045,
    )
    parms["portion"] = parms["step"]/(parms["MTS"]*parms["MW"])*θi[8]
    
    preds = predict(parms, quoteData)
    β = createPredictor(parms, preds, barData)
    trades = getTradeOutcomes(parms, preds, barData, β)
    return runBackTest(parms, trades)
end
function model2(θi::Vector, barData::Dict, preds::Dict)
    # algorithm parameters
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
        "TakerFee"    => 0.0055,
        "MakerFee"    => 0.0045,
    )
    parms["portion"] = parms["step"]/(parms["MTS"]*parms["MW"])*θi[8]
    
    β = createPredictor(parms, preds, barData)
    trades = getTradeOutcomes(parms, preds, barData, β)
    return runBackTest(parms, trades)
end


# Burn in period
burnIn = 1000
θi = [5, 6, 28, 20, 17, 20, 5, 0.453471586]
θhist = zeros(length(θi), burnIn)
θhist[:,1] .= θi
# quoteData, barData = loadData(Minute(θi[1]))

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
    "TakerFee"    => 0.0055,
    "MakerFee"    => 0.0045,
)
parms["portion"] = parms["step"]/(parms["MTS"]*parms["MW"])*θi[8]
# preds = predict(parms, quoteData)
# @save "bot/data/tunBot_2_qutBar.jld2" quoteData barData preds
@load "bot/data/tunBot_2_qutBar.jld2" quoteData barData preds

yhist = zeros(4, burnIn)
yhist[:,1] .= model2(θi, barData, preds)
yi = yhist[1,1]

for i in 11:burnIn
    println("Iter $i")
    θp = copy(θi)
    yp = copy(yi)
    for j in 4:8
        θp[j] = rand(priors[j])
        yp = model2(θp, barData, preds)[1]
        if rand() < (yp/yi)^3
            yi = copy(yp)
            θi = copy(θp)
        end
    end
    θhist[:,i] = copy(θi)
    yhist[:,i] .= model2(θi, barData, preds)
    @save "bot/data/bot1BurnIn_2.jld2" θhist yhist
    display(θi)
    display(yhist[:,i])
    print("\n")
end

println("DONE!!!!")
# Save a copy of the burn in
@save "bot/data/bot1BurnIn_2.jld2" θhist yhist

# ============================================================
# ==================   Analyse posterior   ===================
# ============================================================

# @load "bot/data/bot1BurnIn_2.jld2" θhist yhist
# θ, y = copy(θhist), copy(yhist)
# idx = sortperm(y[2,:], rev = true)
# y[:,idx]

# histogram(θ[1,idx], normalize = :probability, bins = 1:180, title = "tStep")
# histogram(θ[2,idx], normalize = :probability, bins = 1:168, title = "step")
# histogram(θ[3,idx], normalize = :probability, bins = 1:60, title = "trainPeriod")
# histogram(θ[4,idx], normalize = :probability, bins = 1:60, title = "testPeriod")
# histogram(θ[5,idx], normalize = :probability, bins = 1:168, title = "MW")
# histogram(θ[6,idx], normalize = :probability, bins = 1:20, title = "P")
# histogram(θ[7,idx], normalize = :probability, bins = 1:19, title = "MTS")
# histogram(θ[8,idx], normalize = :density, bins = 100, title = "portion")