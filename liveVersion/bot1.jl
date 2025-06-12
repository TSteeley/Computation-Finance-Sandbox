# using FinData, DotEnv
# using JLD2, LinearAlgebra, DataFrames

# DotEnv.load!()

# const coins = ["AAVE", "AVAX", "BAT", "BCH", "BTC", "CRV", "DOGE", "ETH", "GRT", "LINK", "LTC", "MKR", "PEPE",  "SHIB", "SUSHI", "TRUMP", "UNI", "XRP", "XTZ"]

# tStep = Minute(5) # Chunking of quotes
# step = Hour(6) # How often esimates are made on data
# MW = Day(3) # Maximum wait for an order to execute
# trainPeriod = Week(4) # Window size
# testPeriod = Week(2) # Time a trade has to completely finish
# P = 1 # How long back AR looks
# a = 0.75

using Distributions

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

    return mean(y), std(y)
end

function bot1(;P::Int=5)
    @load "bot/data/bot_weights.jld2" β
    @load "liveVersion/bot1Paper/Account.jld2" params

    preds = Dict[]
    println("Predicting prices")
    for c in coins |> ProgressBar
        data = loadCryptoQuotes(c)
        filter!(d -> now()-params["trainPeriod"] < d.t != 0, data)
        filter!(d -> d.bp != 0, data)
        transform!(data, :t => ByRow(t -> round(t, params["tStep"], RoundDown)) => :t)

        TS = combine(groupby(data, :t),
            :bp => last => :bp,
        )
        fillMissing!(TS, step = params["tStep"])
        transform!(TS, :bp => ByRow(log) => :X)

        X, Y = Format_TS_data(TS.X, P, 0)
        μ0, V0 = arPost(X, Y)
        X, Y = Format_TS_data(TS.X, P, 1)
        μ1, V1 = arPost(X, Y)
        X, Y = Format_TS_data(TS.X, P, 2)
        μ2, V2 = arPost(X, Y)
        p = TS.X[end]
        
        TP, BP = hcat(1, μ0, V0, μ1, V1, μ2, V2, p)*β
        push!(preds, Dict(
                "coin" => c,
                "TP" => exp(TP),
                "BP" => exp(BP)
            )
        )
    end
    println("Done")
    idx = sortperm([t["TP"]/t["BP"] for t in preds], rev=true)

    return preds[idx[1:10]]
end

