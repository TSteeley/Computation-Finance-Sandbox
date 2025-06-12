using Distributions, StatsBase
using Plots, CustomPlots
using FinData, Dates, DataFrames
using LinearAlgebra, SpecialFunctions
include("../functions.jl")

# ============================================================
# ==================   Load and Prep Data   ==================
# ============================================================

data = loadCryptoQuotes("BTC")

findall(data.ap .== 0)


d = data[896678:896685,:]

plot(d.t, d.ap)
plot!(d.t, d.bp)
scatter(d.t, d.ap)
scatter!(d.t, d.bp)
