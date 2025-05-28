using FinData, Plots, CustomPlots
using JLD2, Dates

data = loadCryptoQuotes("BTC")
filter!(x -> x.ap != 0, data)

@load "bot/data/basic_1.jld2" θ S

idx = sortperm(S, rev = true)
S = S[idx]
θ = θ[:,idx]

i = 1
trades = Float64[]
while true
    pp = data.ap[i]*0.9975
    sl, tp = data.bp[i] .* exp.(θ[:,rand(1:250)] .* [-1, 1])
    tradeRange = filter(x -> data.t[i] < x.t < data.t[i]+Day(7), data)
    idx = findfirst((tradeRange.bp .< sl) .|| (tradeRange.bp .> tp))
    idx = isnothing(idx) ? size(tradeRange, 1) : idx

    if i+idx == size(data, 1)
        break
    end

    push!(trades, 0.9975*tradeRange.bp[idx] / pp)
    i += idx

    println("$(round(100*(data.t[i]-data.t[1])/(data.t[end]-data.t[1]), digits = 3))% completed")
    println("   Trade earned $(round(tradeRange.bp[idx] / pp, digits = 3))%")
    println("  Score: $(round(prod(trades), digits = 3))\n")
end


plot()