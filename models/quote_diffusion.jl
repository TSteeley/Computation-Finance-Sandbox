using Plots, CustomPlots, FinData, Dates
using Distributions, LinearAlgebra
using Base.Threads, BenchmarkTools

coin = "BTC"

data = loadCryptoQuotes(coin)
filter!(x -> x.ap != 0, data)
n = size(data, 1)

Δt = Day(7)

ul = findlast(data.t .< data.t[end] - 2*Δt)

bp = 0.0
ap = 0.0

for _ in 1:1_000
    r = rand(1:ul)
    idx = findlast(data.t .≤ data.t[r] + Δt)

    bp += ((data.bp[idx] - data.bp[r])/data.bp[r])^2
    ap += ((data.ap[idx] - data.ap[r])/data.ap[r])^2
end

bp /= 1000
ap /= 1000

sqrt(bp)
sqrt(ap)

bp = 0.0
ap = 0.0
@benchmark begin
    r = rand(1:ul)
    idx = findlast(data.t .≤ data.t[r] + Δt)

    global bp, ap
    bp += (data.bp[idx] - data.bp[r])^2
    ap += (data.ap[idx] - data.ap[r])^2
end

bp = 0.0
ap = 0.0
@benchmark begin
    row = data[rand(1:ul),:]
    idx = findlast(data.t .≤ row.t + Δt)

    global bp, ap
    bp += (data.bp[idx] - row.bp)^2
    ap += (data.ap[idx] - row.ap)^2
end