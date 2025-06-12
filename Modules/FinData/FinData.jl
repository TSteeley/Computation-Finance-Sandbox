module FinData
    using HTTP, JSON, DataFrames, TOML
    using Dates, TimeZones, JLD2, ProgressBars

    export fetchCryptoBars, loadCryptoBars, gatherCryptoBars, updateBars
    export fetchCryptoQuotes, loadCryptoQuotes, gatherCryptoQuotes, updateQuotes

    include("CryptoBars.jl")
    include("cryptoQuotes.jl")
end