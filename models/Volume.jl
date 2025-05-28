using Plots, CustomPlots, FinData
using Distributions, LinearAlgebra
using Dates, DataFrames

coin = "ETH"

data = loadCryptoData(coin)

minData = transform(data, :t => ByRow(t -> 60Dates.Hour(t).value + Dates.Minute(t).value) => :t)

minData_grouped = groupby(minData, :t)

minData_g = combine(minData_grouped, 
    :n => sum => :n,
    :v => sum => :v,
)

plot(minData_g[!, :t], minData_g[!, :n], label = "Orders")
plot!(minData_g[!, :t], minData_g[!, :v], label = "Volume")

idx = findall( (data[!,:n] .!= 0) .& (data[!,:v] .!= 0) )
histogram(data[idx,:v] ./ data[idx,:n], bins = 5000, normalise = :pdf, xlims = (0, 3))

g = fit(Gamma, data[idx,:v] ./ data[idx,:n])
fplot!((0, 40), x -> pdf(g, x))
mode(data[idx,:v] ./ data[idx,:n])

Y = data[idx,:v] ./ data[idx,:n]

prior = product_distribution(
    Exponential(1), # α₁
    Exponential(1), # θ₁
    Exponential(1), # α₂
    Exponential(1), # θ₂
    Uniform(0, 1),  # ϵ
)

function f(θ::Vector{Float64})
    α₁, θ₁, α₂, θ₂, ϵ = θ
    G1 = Gamma(α₁, θ₁)
    G2 = Gamma(α₂, θ₂)
    return mean(ϵ * pdf.(G1, Y) + (1 - ϵ) * pdf.(G2, Y))
end


θ = rand(prior)
f(θ)

α₁, θ₁, α₂, θ₂, ϵ = θ
G1 = Gamma(α₁, θ₁)
G2 = Gamma(α₂, θ₂)
f1(x) = ϵ * pdf.(G1, x) + (1 - ϵ) * pdf.(G2, x)

f1(1)

fplot((0,3), f1)
fplot!((0,3), f1)