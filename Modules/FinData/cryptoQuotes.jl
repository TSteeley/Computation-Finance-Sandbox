function getQuotes(query::Dict)
    headers = Dict(
        "accept" => "application/json"
    )
    return HTTP.get(
        "https://data.alpaca.markets/v1beta3/crypto/us/quotes",
        headers,
        query=query
    ).body |> String |> JSON.parse
end

parseTime(t::String) = split(t, ".") |> t -> DateTime(t[1]) + Nanosecond(parse(Int, t[2][1:end-1]))

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
# fetchCryptoQuotes

## Args
 - symbol: e.g. "BTC", "LTC"
 - startTime: Start of period (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ)
 - endTime: End of period (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ)
"""
function fetchCryptoQuotes(symbol::String; startTime::String="", endTime::String="")
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
        getCryptoQuotes(symbol, metaData[symbol]["end"], endTime)
    end

    @load "data/cryptoQuotes/$symbol.jld2" quoteData
    filter!(x -> startTime < x.t < endTime, quoteData)
    return quoteData
end


@doc"""
# getCryptoQuotes

Don't use directly, use fetchCryptoQuotes

"""
function getCryptoQuotes(symbol::String, startTime::DateTime, endTime::DateTime)
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
    
    if .! isnothing(data["next_page_token"])
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
    end
    
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

function gatherCryptoQuotes(symbol::String; startTime::String="", endTime::String="")
    symbol = replace(symbol, r"/USD$" => "")
    
    metaData = TOML.parsefile("data/cryptoQuotes/data.toml")
    
    if symbol ∉ keys(metaData)
        startTime = isempty(startTime) ? error("Data for $(symbol) has not been collected previously. Specify a start time to collect new data.") : DateTime(startTime)
        endTime = isempty(endTime) ? now() : DateTime(endTime)
        getCryptoQuotes(symbol, startTime, endTime)
    else
        startTime = isempty(startTime) ? metaData[symbol]["start"] : DateTime(startTime)
        endTime = isempty(endTime) ? metaData[symbol]["end"] : DateTime(endTime)

        if startTime < metaData[symbol]["start"] # Check if requested period is outside of already saved period
            getCryptoQuotes(symbol, startTime, metaData[symbol]["start"])
        end
    
        if endTime > metaData[symbol]["end"]
            getCryptoQuotes(symbol, metaData[symbol]["end"], endTime)
        end
    end
end

function updateQuotes()
    # println("Updating Quotes")
    for coin in keys(TOML.parsefile("data/cryptoQuotes/data.toml")) |> ProgressBar
        gatherCryptoQuotes(coin, startTime="", endTime=string(now()))
    end
    # println("Done!\n")
end