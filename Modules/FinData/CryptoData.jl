# ============================================================
# ==========   All in one crypto data get function ===========
# ============================================================

@doc"""
# getCryptoData

Do not use directly. Use fetchCryptoData instead.

## Args:
 - symbol: e.g. "BTC", "LTC"
 - startTime: Start of period, defaults to start of day (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ)
 - endTime: End of period, defaults to the current time (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ)
 - limit: max number of data points recieved 1-10_000, defaults to 1_000
"""
function getCryptoData(symbol::String; startTime::String="", endTime::String="", limit::Union{Int, String}=10000) # , timeFrame::String="1T"

    headers = Dict(
        "accept" => "application/json"
    )

    function sendQuery(query::Dict)
        return HTTP.get(
            "https://data.alpaca.markets/v1beta3/crypto/us/bars",
            headers,
            query=query
        ).body |> String |> JSON.parse
    end

    symbol = replace(symbol, r"/USD$" => "")

    metaData = TOML.parsefile("./data/data.toml")

    replace(endTime, r"^(.*T.*)$" => s"\1Z")
    
    query = Dict(
        "symbols"   => symbol*"/USD",
        "timeframe" => "1T",
        "start"     => replace(startTime, r"^(.*T.*[^Z])$" => s"\1Z"),
        "end"       => replace(endTime, r"^(.*T.*[^Z])$" => s"\1Z"),
        "limit"     => limit,
        "sort"      => "asc",
    )

    data = sendQuery(query)

    barData = DataFrames.DataFrame(data["bars"][symbol*"/USD"])
    transform!(barData, :v => ByRow(v -> Float64(v)) => :v)

    while .! isnothing(data["next_page_token"])
        query["page_token"] = data["next_page_token"]
        data = sendQuery(query)

        append!(barData, DataFrames.DataFrame(data["bars"][symbol*"/USD"]))
    end

    transform!(barData, :t => ByRow(t -> ZonedDateTime(t)) => :t)

    if any(readdir("data/") .== "$symbol.jld2") # if some data already saved
        newData = copy(barData)
        @load "data/$symbol.jld2" barData
        append!(barData, newData) |> unique!
    end
    
    sort!(barData, :t)
    @save "data/$symbol.jld2" barData

    metaData[symbol] = Dict(
        "start" => barData[!,:t] |> minimum |> DateTime,
        "end" => barData[!,:t] |> maximum |> DateTime
    )

    open("./data/data.toml", "w") do io
        TOML.print(io, metaData)
    end

    # return barData
end

@doc"""
# fetchCryptoData

Function grabs already saved data or gets more data if not found.

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
function fetchCryptoData(symbol::String; startTime::String="", endTime::String="", timeFrame::String="")

    symbol = replace(symbol, r"/USD" => "")
    metaData = TOML.parsefile("./data/data.toml")

    # Check if we have downloaded data for this symbol before
    if haskey(metaData, symbol)
        startTime = isempty(startTime) ? string(metaData[symbol]["start"]) : startTime
        endTime = isempty(endTime) ? string(metaData[symbol]["end"]) : endTime

        # Ensure all requested data exists
        # If start is before earliest data, get earlier data
        if DateTime(startTime) < metaData[symbol]["start"]
            getCryptoData(symbol, startTime = startTime, endTime = string(metaData[symbol]["start"]))
        end

        # If end is after latest data, update data
        if DateTime(endTime) > metaData[symbol]["end"]
            getCryptoData(symbol, startTime = string(metaData[symbol]["end"]), endTime = endTime)
        end
    else
        startTime = isempty(startTime) ? string(now() - Dates.Month(3)) : startTime
        endTime = isempty(endTime) ? string(now()) : endTime

        getCryptoData(symbol, startTime = startTime, endTime = endTime)
    end
    
    # Load data
    @load "data/$symbol.jld2" barData

    # Keep only in window
    deleteat!(barData, .! (DateTime(startTime) .≤ DateTime.(barData[!,:t]) .≤ DateTime(endTime)))

    # Group by time period
    if timeFrame == "1T" || timeFrame == "1Min" || timeFrame == ""
        return barData
    elseif occursin("T", timeFrame) || occursin("Min", timeFrame)
        n = parse(Int, replace(timeFrame, r"(T|Min|Mins)" => ""))
        T = barData[1,:t]
        transform!(barData, :t => ByRow(t -> T + Dates.Minute(n*floor.(Int, (t - T).value ./ (60000n)))) => :t)
    elseif occursin("H", timeFrame) || occursin("Hour", timeFrame)
        n = parse(Int, replace(timeFrame, r"(H|Hour|Hours)" => ""))
        T = barData[1,:t]
        transform!(barData, :t => ByRow(t -> T + Dates.Hour(n*floor.(Int, (t - T).value ./ (60000n*60)))) => :t)
    elseif occursin("D", timeFrame) || occursin("Day", timeFrame)
        n = parse(Int, replace(timeFrame, r"(D|Day|Days)" => ""))
        T = Date(barData[1,:t])
        transform!(barData, :t => ByRow(t -> T + Dates.Day(n*floor.(Int, (Date(t) - T).value ./ n))) => :t)
    elseif occursin("W", timeFrame) || occursin("Week", timeFrame)
        n = parse(Int, replace(timeFrame, r"(W|Week|Weeks)" => ""))
        T = Date(barData[1,:t])
        transform!(barData, :t => ByRow(t -> T + Dates.Week(n*floor.(Int, (Date(t) - T).value ./ (7n)))) => :t)
    elseif occursin("M", timeFrame) || occursin("Month", timeFrame)
        n = parse(Int, replace(timeFrame, r"(M|Month|Months)" => ""))
        T = Date(barData[1,:t])
        transform!(barData, :t => ByRow(t -> T + Dates.Month(n*floor(Int, ((Month(t) - Month.(T)).value  + 12(Year(t) - Year(T)).value) ./ n))) => :t)
    end
    
    groups = groupby(barData, :t)

    barData_Combined = combine(groups, 
        :v => sum => :v, 
        :c => last => :c,
        :o => first => :o,
        [:vw, :v] => ((vw, v) -> sum(v) == 0 ? 0 : sum(vw .* v)/sum(v)) => :vw,
        :l => minimum => :l,
        :h => maximum => :h,
        :n => sum => :n
    )

    return barData_Combined
end


@doc"""
# loadCryptoData

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
function loadCryptoData(symbol::String; startTime::String="", endTime::String="", timeFrame::String="")

    symbol = replace(symbol, r"/USD" => "")
    metaData = TOML.parsefile("./data/data.toml")

    startTime = isempty(startTime) ? string(metaData[symbol]["start"]) : startTime
    endTime = isempty(endTime) ? string(metaData[symbol]["end"]) : endTime

    # Load data
    @load "data/$symbol.jld2" barData

    # Keep only in window
    deleteat!(barData, .! (DateTime(startTime) .≤ DateTime.(barData[!,:t]) .≤ DateTime(endTime)))

    # Group by time period
    if timeFrame == "1T" || timeFrame == "1Min" || timeFrame == ""
        return barData
    elseif occursin("T", timeFrame) || occursin("Min", timeFrame)
        n = parse(Int, replace(timeFrame, r"(T|Min|Mins)" => ""))
        T = barData[1,:t]
        transform!(barData, :t => ByRow(t -> T + Dates.Minute(n*floor.(Int, (t - T).value ./ (60000n)))) => :t)
    elseif occursin("H", timeFrame) || occursin("Hour", timeFrame)
        n = parse(Int, replace(timeFrame, r"(H|Hour|Hours)" => ""))
        T = barData[1,:t]
        transform!(barData, :t => ByRow(t -> T + Dates.Hour(n*floor.(Int, (t - T).value ./ (60000n*60)))) => :t)
    elseif occursin("D", timeFrame) || occursin("Day", timeFrame)
        n = parse(Int, replace(timeFrame, r"(D|Day|Days)" => ""))
        T = Date(barData[1,:t])
        transform!(barData, :t => ByRow(t -> T + Dates.Day(n*floor.(Int, (Date(t) - T).value ./ n))) => :t)
    elseif occursin("W", timeFrame) || occursin("Week", timeFrame)
        n = parse(Int, replace(timeFrame, r"(W|Week|Weeks)" => ""))
        T = Date(barData[1,:t])
        transform!(barData, :t => ByRow(t -> T + Dates.Week(n*floor.(Int, (Date(t) - T).value ./ (7n)))) => :t)
    elseif occursin("M", timeFrame) || occursin("Month", timeFrame)
        n = parse(Int, replace(timeFrame, r"(M|Month|Months)" => ""))
        T = Date(barData[1,:t])
        transform!(barData, :t => ByRow(t -> T + Dates.Month(n*floor(Int, ((Month(t) - Month.(T)).value  + 12(Year(t) - Year(T)).value) ./ n))) => :t)
    end
    
    groups = groupby(barData, :t)

    barData_Combined = combine(groups, 
        :v => sum => :v, 
        :c => last => :c,
        :o => first => :o,
        [:vw, :v] => ((vw, v) -> sum(v) == 0 ? 0 : sum(vw .* v)/sum(v)) => :vw,
        :l => minimum => :l,
        :h => maximum => :h,
        :n => sum => :n
    )

    return barData_Combined
end