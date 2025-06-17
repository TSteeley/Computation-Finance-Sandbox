#=
    v  => Volume
    c  => Close
    o  => Open
    t  => Time
    vw => Volume weighted average
    l  => Low
    h  => High
    n  => Number of orders
=#

using HTTP, JSON, DataFrames, TOML
using Dates, TimeZones, JLD2

# ============================================================
# ===================   Helper Functions   ===================
# ============================================================


# Added a constructor for TimePeriod. Give a string and it returns a time period with the most specific type.
TIME_UNITS = Dict(
    r"^(ns|nanosecond)" => Nanosecond,
    r"^(us|microsecond)" => Microsecond,
    r"^(ms|millisecond)" => Millisecond,
    r"^(S|Second)$" => Second,
    r"^(T|Min)$" => Minute,
    r"^(H|Hour)$" => Hour,
    r"^(D|Day)$" => Day,
    r"^(W|Week)$" => Week,
    r"^(M|Month)$" => Month,
    r"^(Y|Year)$" => Year
)
@doc"""
# Period

Convert a string to a valid Period. Type of output is most specific type which is a sub-type of Period.

## Valid periods;
- ns, nanosecond  => Nanosecond
- us, microsecond => Microsecond
- ms, millisecond => Millisecond
- S,  Second      => Second
- T,  Min         => Minute
- H,  Hour        => Hour
- D,  Day         => Day
- W,  Week        => Week
- M,  Month       => Month
- Y,  Year        => Year
"""
function Period(str::String)
    m = match(r"^(\d+)\s*([A-Za-z]+)$", strip(str))
    m === nothing && error("Invalid format: $str")

    n = parse(Int, m.captures[1])
    suffix = m.captures[2]

    for (pattern, func) in TIME_UNITS
        if occursin(pattern, suffix)
            return func(n)
        end
    end
    error("$str not recognised")
end

# ============================================================
# =====================   Test values   ======================
# ============================================================

symbol = "BTC"
startTime = "2023-01-01"
endTime = ""
timeFrame = "1T"


# ============================================================
# ========================   Setup   =========================
# ============================================================

function sendBarQuery(query::Dict)
    return HTTP.get(
        "https://data.alpaca.markets/v1beta3/crypto/us/bars",
        Dict("accept" => "application/json"),
        query=query
    ).body |> String |> JSON.parse
end

# ============================================================
# ==========   All in one crypto data get function ===========
# ============================================================

parseBarTime(str::String) = DateTime(str[1:end-1])

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
function loadCryptoBars(symbol::String; startTime::String="", endTime::String="", timeFrame::String="")
    symbol = replace(symbol, r"/USD" => "")
    metaData = TOML.parsefile("data/cryptoBars/data.toml")

    startTime = isempty(startTime) ? string(metaData[symbol]["start"]) : startTime
    endTime = isempty(endTime) ? string(metaData[symbol]["end"]) : endTime

    # Load data
    @load "data/cryptoBars/$symbol.jld2" barData

    if isempty(timeFrame)
        return barData
    else
        timeFrame = Period(timeFrame)
        # Keep only in window
        transform!(barData, :t => ByRow(t -> round(t, timeFrame, RoundDown)) => :t)
        filter!(x -> startTime .≤ x.t .≤ endTime, barData)

        # Group by time period
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
end

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


# loadCryptoBars("BTC", timeStep = "5T")
# loadCryptoBars("BTC", timeFrame=Minute(5))

# ============================================================
# =======================   testing   ========================
# ============================================================

validPeriods = ["T","Min","Mins","H","Hour","Hours","D","Day","Days","W","Week","Weeks","M","Month","Months"]

tests = ["$(rand(1:60))$(rand(validPeriods))" for _ in 1:100]



timeFrame = "5D"

str = "1 second"

m = match(r"^(\d+)\s*([A-Za-z]+)$", strip(str))

n = parse(Int, m.captures[1])
suffix = m.captures[2]

for (pattern, func) in TIME_UNITS
    if occursin(pattern, suffix)
        println(func(n))
    end
end


if occursin("T", tstep) || occursin("Min", tstep)
    n = parse(Int, replace(tstep, r"(T|Min|Mins)" => ""))
    T = barData[1,:t]
    transform!(barData, :t => ByRow(t -> T + Dates.Minute(n*floor.(Int, (t - T).value ./ (60000n)))) => :t)
elseif occursin("H", tstep) || occursin("Hour", tstep)
    n = parse(Int, replace(tstep, r"(H|Hour|Hours)" => ""))
    T = barData[1,:t]
    transform!(barData, :t => ByRow(t -> T + Dates.Hour(n*floor.(Int, (t - T).value ./ (60000n*60)))) => :t)
elseif occursin("D", tstep) || occursin("Day", tstep)
    n = parse(Int, replace(tstep, r"(D|Day|Days)" => ""))
    T = Date(barData[1,:t])
    transform!(barData, :t => ByRow(t -> T + Dates.Day(n*floor.(Int, (Date(t) - T).value ./ n))) => :t)
elseif occursin("W", tstep) || occursin("Week", tstep)
    n = parse(Int, replace(tstep, r"(W|Week|Weeks)" => ""))
    T = Date(barData[1,:t])
    transform!(barData, :t => ByRow(t -> T + Dates.Week(n*floor.(Int, (Date(t) - T).value ./ (7n)))) => :t)
elseif occursin("M", tstep) || occursin("Month", tstep)
    n = parse(Int, replace(tstep, r"(M|Month|Months)" => ""))
    T = Date(barData[1,:t])
    transform!(barData, :t => ByRow(t -> T + Dates.Month(n*floor(Int, ((Month(t) - Month.(T)).value  + 12(Year(t) - Year(T)).value) ./ n))) => :t)
end

metaData = TOML.parsefile("data/cryptoBars/data.toml")

coins = ["AAVE", "AVAX", "BAT", "BCH", "BTC", "CRV", "DOGE", "DOT", "ETH", "GRT", "LINK", "LTC", "MKR", "PEPE", "SHIB", "SOL", "SUSHI", "TRUMP", "UNI", "XRP", "XTZ", "YFI"]

for coin in coins |> ProgressBar
    data = loadCryptoBars(coin)
    metaData[coin]["end"] = data.t[end]
end

open("data/cryptoBars/data.toml", "w") do io
    TOML.print(io, metaData)
end

endTime = now(UTC)
for coin in coins |> ProgressBar
    getCryptoBars(coin, DateTime("2025-05-01"), endTime)
end