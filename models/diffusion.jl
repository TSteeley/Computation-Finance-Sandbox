using Plots, CustomPlots, FinData
using Distributions, LinearAlgebra
using Base.Threads

T = floor.(Int, range(1, 50_000, length = 100))
V = zeros(length(T))

coin = "ETH"

fetchCryptoData(coin, startTime="2023-01-01")

for (i, t) in enumerate(T) |> collect
    data = loadCryptoData(coin, timeFrame="$(t)T")
    V[i] = data[2:end, :o] - data[1:end-1, :o] |> x -> x'*x / length(x)
    # V[i] = (data[2:end, :o] - data[1:end-1, :o]) ./ data[1:end-1, :o] |> x -> x'*x/length(x)
    # V[i] = (data[1:end-1, :o] - data[2:end, :o]) ./ data[1:end-1, :o] |> var
end

f = plot(T, V, xlabel = "Time (Minutes)", ylabel = "Variance", title = "Diffusion vs Time in $coin Data", label = "")

m = inv(T[1:40]'*T[1:40])*T[1:40]'*V[1:40]
fplot!(f, T, t -> m*t, label = "")

savefig(f, "figures/AnnomDiffusion_$coin.pdf")

# AAVE, AVAX, BAT, BCH, BTC, CRV, DOGE, DOT, ETH, GRT, LINK, LTC, MKR, PEPE, SHIB, SOL, SUSHI, TRUMP, UNI, USDC, USDT, XRP, XTZ, YFI

data = loadCryptoData(coin, timeFrame="$(t)T")
V[i] = (data[2:end, :o] - data[1:end-1, :o]) ./ data[1:end-1, :o] |> x -> x'*x/length(x)

log_open = log.(data[!,:o])

idx = findall((data[2:end, :t] - data[1:end-1, :t]).value .== 60_000)

data[2:end, :t] - data[1:end-1, :t]

histogram(log_open[2:end] - log_open[1:end-1], bins = 5000, normalise=:pdf, xlims=0.005*[-1, 1])