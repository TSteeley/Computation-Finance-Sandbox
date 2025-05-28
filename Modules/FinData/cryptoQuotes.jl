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

symbol = "BTC/USD"
startTime = "2023-01-01"
endTime = ""
pageToken = ""
sort = "asc"

function getCryptoQuotes(symbol::String; startTime::String="", endTime::String="")
    symbol = replace(symbol, r"/USD$" => "")
    
    metaData = TOML.parsefile("./quotes/data.toml")
    
    query = Dict(
        "symbols"    => symbol*"/USD",
        "start"      => replace(startTime, r"^(.*T.*[^Z])$" => s"\1Z"),
        "end"        => replace(endTime, r"^(.*T.*[^Z])$" => s"\1Z"),
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
    
    if any(readdir("quotes/") .== "$symbol.jld2") # if some data already saved
        newData = copy(quoteData)
        @load "quotes/$symbol.jld2" quoteData
        append!(quoteData, newData) |> unique!
    end
    
    sort!(quoteData, :t)
    @save "quotes/$symbol.jld2" quoteData
    
    metaData[symbol] = Dict(
        "start" => quoteData[!,:t] |> minimum,
        "end" => quoteData[!,:t] |> maximum
    )
    
    open("quotes/data.toml", "w") do io
        TOML.print(io, metaData)
    end
end

function loadCryptoQuotes(symbol::String; startTime::String="", endTime::String="")
    symbol = replace(symbol, r"/USD$" => "")
    
    metaData = TOML.parsefile("./quotes/data.toml")
    
    @load "quotes/$symbol.jld2" quoteData
    
    if isempty(startTime) && isempty(endTime)
        return quoteData
    end

    startTime = isempty(startTime) ? metaData[symbol]["start"] : DatTime(startTime)
    endTime = isempty(endTime) ? metaData[symbol]["end"] : DateTime(endTime)

    filter!(x -> startTime < x.t < endTime, quoteData)
    return quoteData
end