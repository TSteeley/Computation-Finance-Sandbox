#=
    t  => Time
    ap => Ask Price
    as => Ask Size
    bp => Bid Price
    bs => Bid Size
=#

using HTTP, JSON, DataFrames, TOML
using Dates, TimeZones, JLD2
using CustomPlots, Plots

headers = Dict(
    "accept" => "application/json"
)

function getQuotes(query::Dict)
    return HTTP.get(
        "https://data.alpaca.markets/v1beta3/crypto/us/quotes",
        headers,
        query=query
    ).body |> String |> JSON.parse
end

parseTime(t::String) = split(t, ".") |> t -> DateTime(t[1]) + Nanosecond(parse(Int, t[2][1:end-1]))

symbol = "BTC"
startTime = "2023-01-01"
endTime = ""

function loadCryptoQuotes(symbol::String)
    symbol = replace(symbol, r"/USD$" => "")
    try
        @load "data/cryptoQuotes/$symbol.jld2" quoteData
        return quoteData
    catch
        error("Data for $(coin) has not been collected previously. Specify startTime to collect new data.")
    end
end


@doc"""
# loadCryptoQuotes

## Args
 - symbol: e.g. "BTC", "LTC"
 - startTime: Start of period (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ)
 - endTime: End of period (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ)
"""
function loadCryptoQuotes(symbol::String; startTime::String="", endTime::String="")
    symbol = replace(symbol, r"/USD$" => "")
    
    metaData = TOML.parsefile("data/cryptoQuotes/data.toml")
    
    if symbol ∉ keys(metaData)
        startTime = isempty(startTime) ? error("Data for $(symbol) has not been collected previously. Specify a start time to collect new data.") : DateTime(startTime)
        endTime = isempty(endTime) ? now() : DateTime(endTime)
        getCryptoQuotes(symbol, startTime, endTime)
        @load "data/cryptoQuotes/$symbol.jld2" quoteData
        return quoteData
    end
    
    startTime = isempty(startTime) ? metaData[symbol]["start"] : DateTime(startTime)
    endTime = isempty(endTime) ? metaData[symbol]["end"] : DateTime(endTime)

    if startTime < metaData[symbol]["start"] # Check if requested period is outside of already saved period
        getCryptoQuotes(symbol, startTime, metaData[symbol]["start"])
    end

    if endTime > metaData[symbol]["end"]
        getCryptoData(symbol, metaData[symbol]["end"], endTime)
    end

    @load "data/cryptoQuotes/$symbol.jld2" quoteData
    filter!(x -> startTime < x.t < endTime, quoteData)
    return quoteData
end


@doc"""
# getCryptoQuotes

Don't use directly, use fetchCryptoQuotes

"""
function getCryptoQuotes(symbol::String, startTime::Date, endTime::Date)
    metaData = TOML.parsefile("data/cryptoQuotes/data.toml")
    
    query = Dict(
        "symbols"    => symbol*"/USD",
        "start"      => replace(string(startTime), r"^(.*T.*[^Z])$" => s"\1Z"),
        "end"        => replace(string(endTime), r"^(.*T.*[^Z])$" => s"\1Z"),
        "limit"      => 10_000,
        "page_token" => "",
        "sort"       => "asc",
    )
    
    data = getQuotes(query)
    
    quoteData = DataFrames.DataFrame(data["quotes"][symbol*"/USD"])
    
    while .! isnothing(data["next_page_token"])
        query["page_token"] = data["next_page_token"]
        data = getQuotes(query)
    
        append!(quoteData, DataFrames.DataFrame(data["quotes"][symbol*"/USD"]))
    end
    
    transform!(quoteData, :t => ByRow(parseTime) => :t)
    
    if "$symbol.jld2" ∈ readdir("data/cryptoQuotes/") # if some data already saved
        newData = copy(quoteData)
        @load "data/cryptoQuotes/$symbol.jld2" quoteData
        append!(quoteData, newData) |> unique!
    end
    
    sort!(quoteData, :t)
    @save "data/cryptoQuotes/$symbol.jld2" quoteData
    
    if symbol ∈ keys(metaData)
        metaData[symbol] = Dict(
            "start" => min(startTime, metaData[symbol]["start"]),
            "end" => max(endTime, metaData[symbol]["end"])
        )
    else
        metaData[symbol] = Dict(
            "start" => startTime,
            "end" => endTime,
        )
    end
    
    open("data/cryptoQuotes/data.toml", "w") do io
        TOML.print(io, metaData)
    end
end

data = quoteData[500:510,:]

p = plot()
plot!(p, data.t, data.ap, lc = :green, label = "Ask Price")
plot!(p, data.t, data.bp, lc = :red, label = "Bid Price")
scatter!(p, data.t, data.ap, mc = "green", label = "")
scatter!(p, data.t, data.bp, mc = "red", label = "")

getCryptoQuotes("BTC", startTime = "2023-01-01", endTime = "2025-05-01")


