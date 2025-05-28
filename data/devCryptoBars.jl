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
using CustomPlots
include("../functions.jl")

# ============================================================
# =====================   Test values   ======================
# ============================================================

symbol = "BTC/USD"
startTime = ""
endTime = ""
limit = 10000
tstep = Minute(1)


# ============================================================
# ========================   Setup   =========================
# ============================================================

const headers = Dict(
    "accept" => "application/json"
)

function sendQuery(query::Dict)
    return HTTP.get(
        "https://data.alpaca.markets/v1beta3/crypto/us/bars",
        headers,
        query=query
    ).body |> String |> JSON.parse
end

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
function getCryptoData(symbol::String; startTime::Union{String, DateTime}="", endTime::Union{String, DateTime}="", limit::Union{Int, String}=10000)

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
 - timeframe: Dates TimePeriod or
    - [1-59]Min or [1-59]T; by minute aggregation
    - [1-23]Hour or [1-23]H; by hour aggregation
    - 1Day or 1D; by day aggregation
    - 1Week or 1W; by week aggregation
    - [1,2,3,4,6,12]Month or [1,2,3,4,6,12]M; by month aggregation

"""
function fetchCryptoData(symbol::String; startTime::Union{String, DateTime}="", endTime::Union{String, DateTime}="", step::Union{String, TimePeriod}=Minute(1))

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

    typeof(tstep) == String

    # Group by time period
    if tstep == "1T" || tstep == "1Min" || tstep == ""
        return barData
    elseif occursin("T", tstep) || occursin("Min", tstep)
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
    end
    
    tstep = TimePeriod(timeFrame)
    transform!(barData, :t => ByRow(t -> round(t, tstep, RoundDown)) => :t)
    
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

# ============================================================
# =======================   testing   ========================
# ============================================================

validPeriods = ["T","Min","Mins","H","Hour","Hours","D","Day","Days","W","Week","Weeks","M","Month","Months"]

tests = ["$(rand(1:60))$(rand(validPeriods))" for _ in 1:100]

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
function TimePeriod(str::String)
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