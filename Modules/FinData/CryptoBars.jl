# ============================================================
# ==========   All in one crypto data get function ===========
# ============================================================

parseBarTime(str::String) = DateTime(str[1:end-1])

const headers = Dict(
    "accept" => "application/json"
)

function sendBarQuery(query::Dict)
    return HTTP.get(
        "https://data.alpaca.markets/v1beta3/crypto/us/bars",
        headers,
        query=query
    ).body |> String |> JSON.parse
end
@doc"""
# getCryptoBars

Do not use directly. Use fetchCryptoBars instead.

## Args:
 - symbol: e.g. "BTC", "LTC"
 - startTime: Start of period, defaults to start of day (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ)
 - endTime: End of period, defaults to the current time (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ)
"""
function getCryptoBars(symbol::String, startTime::DateTime, endTime::DateTime)
    symbol = replace(symbol, r"/USD$" => "")
    metaData = TOML.parsefile("data/cryptoBars/data.toml")
    
    query = Dict(
        "symbols"   => symbol*"/USD",
        "timeframe" => "1T",
        "start"     => replace(string(startTime), r"^(.*T.*[^Z])$" => s"\1Z"),
        "end"       => replace(string(endTime), r"^(.*T.*[^Z])$" => s"\1Z"),
        "limit"     => 10_000,
        "sort"      => "asc",
    )

    data = sendBarQuery(query)
    .! isempty(data["bars"]) || return
    barData = DataFrames.DataFrame(data["bars"][symbol*"/USD"])
    barData.v = Float64.(barData.v)

    while .! isnothing(data["next_page_token"])
        query["page_token"] = data["next_page_token"]
        data = sendBarQuery(query)

        append!(barData, DataFrames.DataFrame(data["bars"][symbol*"/USD"]))
    end
    transform!(barData, :t => ByRow(parseBarTime) => :t)

    if "$symbol.jld2" ∈ readdir("data/cryptoBars/") # if some data already saved
        newData = copy(barData)
        @load "data/cryptoBars/$symbol.jld2" barData
        append!(barData, newData)
    end
    
    unique!(barData)
    sort!(barData, :t)
    @save "data/cryptoBars/$symbol.jld2" barData
    
    if symbol ∈ keys(metaData)
        metaData[symbol] = Dict(
            "start" => min(startTime, metaData[symbol]["start"]),
            "end" => max(barData.t[end], metaData[symbol]["end"])
        )
    else
        metaData[symbol] = Dict(
            "start" => startTime,
            "end" => barData.t[end],
        )
    end

    open("data/cryptoBars/data.toml", "w") do io
        TOML.print(io, metaData)
    end
end

@doc"""
# fetchCryptoBars

Function grabs already saved data or gets more data if not found.

## Args:
 - symbol: e.g. "BTC", "LTC"
 - startTime: Start of period, defaults to start of day (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ)
 - endTime: End of period, defaults to the current time (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ)
 - timeFrame: Dates TimePeriod or
    - [1-59]Min or [1-59]T; by minute aggregation
    - [1-23]Hour or [1-23]H; by hour aggregation
    - 1Day or 1D; by day aggregation
    - 1Week or 1W; by week aggregation
    - [1,2,3,4,6,12]Month or [1,2,3,4,6,12]M; by month aggregation

"""
function fetchCryptoBars(symbol::String; startTime::String="", endTime::String="", timeFrame::T=Minute(1)) where T <: Period
    symbol = replace(symbol, r"/USD" => "")
    metaData = TOML.parsefile("data/cryptoBars/data.toml")

    # Check if we have downloaded data for this symbol before
    if symbol ∉ keys(metaData)
        startTime = isempty(startTime) ? error("Data for $(symbol) has not been collected previously. Specify a start time to collect new data.") : DateTime(startTime)
        endTime = isempty(endTime) ? now() : DateTime(endTime)
        getCryptoBars(symbol, startTime, endTime)
    elseif isempty(startTime) && isempty(endTime)
        @load "data/cryptoBars/$symbol.jld2" barData
        return barData
    else
        startTime = isempty(startTime) ? metaData[symbol]["start"] : DateTime(startTime)
        endTime = isempty(endTime) ? metaData[symbol]["end"] : DateTime(endTime)

        if startTime < metaData[symbol]["start"]
            getCryptoBars(symbol, startTime, metaData[symbol]["start"])
        end
        if endTime > metaData[symbol]["end"]
            getCryptoBars(symbol, metaData[symbol]["end"], endTime)
        end
    end
    
    # Load data
    @load "data/cryptoBars/$symbol.jld2" barData

    # Keep only in window
    filter!(x -> startTime < x.t < endTime, barData)

    # Group by time period
    transform!(barData, :t => ByRow(t -> round(t, timeFrame, RoundDown)) => :t)
    return combine(groupby(barData, :t),
        :v => sum => :v, 
        :c => last => :c,
        :o => first => :o,
        [:vw, :v] => ((vw, v) -> sum(v) == 0 ? 0 : sum(vw .* v)/sum(v)) => :vw, # Approximate VWA
        :l => minimum => :l,
        :h => maximum => :h,
        :n => sum => :n
    )
end


@doc"""
# loadCryptoBars

Function only grabs already saved data.

Will not find new data.

## Args:
 - symbol: e.g. "BTC", "LTC"
 - startTime: Start of period, defaults to start of day (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ)
 - endTime: End of period, defaults to the current time (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ)
 - timeframe: 
    - [1-59]Min or [1-59]T; by minute aggregation
    - [1-23]Hour or [1-23]H; by hour aggregation
    - 1Day or 1D; by day aggregation
    - 1Week or 1W; by week aggregation
    - [1,2,3,4,6,12]Month or [1,2,3,4,6,12]M; by month aggregation

"""
function loadCryptoBars(symbol::String; timeFrame::T=Minute(1)) where T <: Period
    symbol = replace(symbol, r"/USD" => "")
    @load "data/cryptoBars/$symbol.jld2" barData
    if timeFrame == Minute(1)
        return barData
    end

    transform!(barData, :t => ByRow(t -> round(t, timeFrame, RoundDown)) => :t)
    return combine(groupby(barData, :t),
        :v => sum => :v, 
        :c => last => :c,
        :o => first => :o,
        [:vw, :v] => ((vw, v) -> sum(v) == 0 ? 0 : sum(vw .* v)/sum(v)) => :vw, # Approximate VWA
        :l => minimum => :l,
        :h => maximum => :h,
        :n => sum => :n
    )
end

# function loadCryptoBars(symbol::String; startTime::String="", endTime::String="", timeFrame::String="")
#     symbol = replace(symbol, r"/USD" => "")
#     metaData = TOML.parsefile("data/cryptoBars/data.toml")

#     startTime = isempty(startTime) ? string(metaData[symbol]["start"]) : startTime
#     endTime = isempty(endTime) ? string(metaData[symbol]["end"]) : endTime

#     # Load data
#     @load "data/cryptoBars/$symbol.jld2" barData

#     # Keep only in window
#     transform!(barData, :t => ByRow(t -> round(t, timeFrame, RoundDown)) => :t)

#     # Group by time period
#     return combine(groupby(barData, :t),
#         :v => sum => :v, 
#         :c => last => :c,
#         :o => first => :o,
#         [:vw, :v] => ((vw, v) -> sum(v) == 0 ? 0 : sum(vw .* v)/sum(v)) => :vw, # Approximate VWA
#         :l => minimum => :l,
#         :h => maximum => :h,
#         :n => sum => :n
#     )
# end

function gatherCryptoBars(symbol::String; startTime::String="", endTime::String="")
    symbol = replace(symbol, r"/USD" => "")
    metaData = TOML.parsefile("data/cryptoBars/data.toml")

    # Check if we have downloaded data for this symbol before
    if symbol ∉ keys(metaData)
        startTime = isempty(startTime) ? error("Data for $(symbol) has not been collected previously. Specify a start time to collect new data.") : DateTime(startTime)
        endTime = isempty(endTime) ? now() : DateTime(endTime)
        getCryptoBars(symbol, startTime, endTime)
    else
        startTime = isempty(startTime) ? metaData[symbol]["start"] : DateTime(startTime)
        endTime = isempty(endTime) ? metaData[symbol]["end"] : DateTime(endTime)

        if startTime < metaData[symbol]["start"]
            getCryptoBars(symbol, startTime, metaData[symbol]["start"])
        end
        if endTime > metaData[symbol]["end"]
            getCryptoBars(symbol, metaData[symbol]["end"], endTime)
        end
    end
end

function updateBars()
    # println("Updating Bars")
    for coin in keys(TOML.parsefile("data/cryptoBars/data.toml")) |> ProgressBar
        gatherCryptoQuotes(coin, startTime="", endTime=string(now(UTC)))
    end
    # println("Done!")
end