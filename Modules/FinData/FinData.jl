module FinData
    using HTTP, JSON, DataFrames, TOML
    using Dates, TimeZones, JLD2

    export fetchCryptoData, loadCryptoData
    export getCryptoQuotes, loadCryptoQuotes

    include("CryptoData.jl")
    include("cryptoQuotes.jl")
end