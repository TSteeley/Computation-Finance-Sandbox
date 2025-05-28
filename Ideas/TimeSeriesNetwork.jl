using Plots, CustomPlots, ProgressBars
using FinData, Dates, DataFrames
using LinearAlgebra, Flux
using SpecialFunctions
include("../int.jl")
include("../functions.jl")

# ============================================================
# ==================   Load and Prep Data   ==================
# ============================================================

tStep = Minute(30)
data = loadCryptoQuotes("BTC")
transform!(data, :t => ByRow(t -> round(t, tStep, RoundDown)) => :t)

minData = combine(groupby(data, :t),
    :bp => mean => :bp,
    # :ap => mean => :ap,
)

trainPeriod = Week(3)
testPeriod = Week(1)

r = sample(minData.t[1], minData.t[end]-(trainPeriod+testPeriod+Week(1)), step = tStep)
train = filter(x -> r .≤ x.t .< r+trainPeriod, minData)
test = filter(x -> r+trainPeriod .≤ x.t .< r+trainPeriod+testPeriod, minData)

fillMissing!(train, step = tStep)
fillMissing!(test, step = tStep)

# ============================================================
# =========================   Loss   =========================
# ============================================================

