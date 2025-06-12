using FinData, DotEnv, ProgressBars, HTTP, JSON
using JLD2, LinearAlgebra, DataFrames, CSV
include("../functions.jl")
include("bot1.jl")
DotEnv.load!()

const coins = ["AAVE", "AVAX", "BAT", "BCH", "BTC", "CRV", "DOGE", "ETH", "GRT", "LINK", "LTC", "MKR", "PEPE",  "SHIB", "SUSHI", "TRUMP", "UNI", "XRP", "XTZ"]

parseTime(t::String) = split(t, ".") |> t -> DateTime(t[1]) + Nanosecond(parse(Int, t[2][1:end-1]))

const header = Dict(
    "accept" => "application/json",
    "content-type" => "application/json",
    "APCA-API-KEY-ID" => ENV["PAPER_KEY"],
    "APCA-API-SECRET-KEY" => ENV["PAPER_SECRET_KEY"],
)
const getHeader = Dict(
    "accept" => "application/json",
    "APCA-API-KEY-ID" => ENV["PAPER_KEY"],
    "APCA-API-SECRET-KEY" => ENV["PAPER_SECRET_KEY"],
)

function getCurrentPrice(coin::String)
    req = HTTP.get(
        "https://data.alpaca.markets/v1beta3/crypto/us/latest/bars",
        Dict("accept" => "application/json"),
        query=Dict(
            "symbols" => coin*"/USD"
        )
    ).body |> String |> JSON.parse
    return req["bars"][coin*"/USD"]
end

function placeOrder(pred::Dict)
    @load "liveVersion/bot1Paper/Account.jld2" Account params
    Log = CSV.read("liveVersion/bot1Paper/tradeLog.csv", DataFrame)
    if Account["Liquidity"] > 1
        query = Dict{String, Any}()
        p = getCurrentPrice(pred["coin"])["c"]
        orderVal = min(Account["Value"]*params["portion"],Account["Liquidity"])
        if pred["BP"] > p # market order
            query = Dict(
                "symbol" => pred["coin"]*"/USD",
                "qty" => orderVal/p,
                "time_in_force" => "gtc",
                "side" => "buy",
                "type" => "market",
                "order_class" => "simple",
                "position_intent" => "buy_to_open",
            )
        else
            query = Dict(
                "symbol" => pred["coin"]*"/USD",
                "qty" => orderVal/pred["BP"],
                "side" => "buy",
                "type" => "limit",
                "time_in_force" => "gtc",
                "limit_price" => pred["BP"], # Only if limit order
                "order_class" => "simple",
                "position_intent" => "buy_to_open",
            )
        end

        req = HTTP.post(
            ENV["PAPER_ENDPOINT"]*"/orders",
            header,
            JSON.json(query)
        ).body |> String |> JSON.parse

        append!(Log, 
            Dict(
                "coin"         => pred["coin"],
                "createdAt"    => round(now(), Hour(6), RoundDown),
                "BP"           => pred["BP"],
                "TP"           => pred["TP"],
                "PL_pc"        => 0.0,
                "orderID"      => req["id"],
                "status"       => "order open",
                "value"        => orderVal,
                "orderType"    => req["order_type"],
                "orderQty"     => parse(Float64, req["qty"]),
                "avgFillPrice" => 0.0,
                "avgSellPrice" => 0.0,
                "sellQty"      => 0.0,
                "sellID"       => "",
                "sellType"     => "",
                "totalFees"    => 0.0,
                "buyFee"       => 0.0,
                "sellFee"      => 0.0,
                "PL_dollars"   => 0.0,
                "completeTime" => DateTime(0),
            )
        )
        Account["Liquidity"] -= orderVal
        @save "liveVersion/bot1Paper/Account.jld2" Account params
        CSV.write("liveVersion/bot1Paper/tradeLog.csv", Log)
    end
end

@doc"""
# updateOrders

Gets all known unfilled orders, and acts accordingly. If they are too old they are killed, if they are filled the order is updated, otherwise left alone.
"""
function updateOrders()
    @load "liveVersion/bot1Paper/Account.jld2" Account params
    Log = CSV.read("liveVersion/bot1Paper/tradeLog.csv", DataFrame)

    Orders = @view Log[findall(Log[!,"status"] .== "order open"),:]
    
    for (i, ord) in enumerate(eachrow(Orders))
        req = HTTP.get(
            ENV["PAPER_ENDPOINT"]*"/orders/"*ord["orderID"],
            getHeader
        ).body |> String |> JSON.parse
        if req["filled_at"] !== nothing
            Orders[i, "status"] = "trade active"
            Orders[i, "orderQty"] = parse(Float64, req["filled_qty"])
            Orders[i, "avgFillPrice"] = parse(Float64, req["filled_avg_price"])
            fee = Orders[i, "orderQty"]*(ord["orderType"] == "limit" ? params["TakerFee"] : params["MakerFee"])
            Orders[i, "buyFee"] = fee*Orders[i, "avgFillPrice"]
            Orders[i, "totalFees"] += fee*Orders[i, "avgFillPrice"]

            # Update order value and liquidity in case there is a discrepancy between intended and actual trade value
            value = Orders[i, "avgFillPrice"]*Orders[i, "orderQty"]
            Account["Liquidity"] += Orders[i,"value"] - value
            Orders[i,"value"] = value

            # Place order to close
            query = Dict(
                "symbol" => ord["coin"]*"/USD",
                "qty" => Orders[i, "orderQty"]-fee,
                "side" => "sell",
                "type" => "limit",
                "time_in_force" => "gtc",
                "limit_price" => ord["TP"],
                "order_class" => "simple",
                "position_intent" => "sell_to_close",
            )
            req2 = HTTP.post(
                ENV["PAPER_ENDPOINT"]*"/orders",
                header,
                JSON.json(query)
            ).body |> String |> JSON.parse

            # Update info
            Orders[i,"sellID"] = req2["id"]
            Orders[i,"sellQty"] = parse(Float64, req["qty"])
            Orders[i,"sellType"] = "limit"
        elseif ord[:createdAt] + params["MW"] < now()
            Orders[i,"status"] = "order cancelled"
            HTTP.delete(
                ENV["PAPER_ENDPOINT"]*"/orders/"*ord["orderID"],
                getHeader
            )
            # Return unused liquidity
            Account["Liquidity"] += ord["value"]
            Trades[i, "completeTime"] = now()
        end
        # Saves every loop in case an error breaks it
        @save "liveVersion/bot1Paper/Account.jld2" Account params
        CSV.write("liveVersion/bot1Paper/tradeLog.csv", Log)
    end
end

function updateTrades()
    @load "liveVersion/bot1Paper/Account.jld2" Account params
    Log = CSV.read("liveVersion/bot1Paper/tradeLog.csv", DataFrame)

    Trades = @view Log[findall(Log[!,"status"] .== "trade active"),:]

    for (i, trade) in enumerate(eachrow(Trades))
        req = HTTP.get(
            ENV["PAPER_ENDPOINT"]*"/orders/"*trade["sellID"],
            getHeader
        ).body |> String |> JSON.parse
        if req["filled_at"] !== nothing # Trade has executed, update info
            Trades[i, "status"] = "trade closed"
            Trades[i, "sellQty"] = parse(Float64, req["filled_qty"])
            Trades[i, "avgSellPrice"] = parse(Float64, req["filled_avg_price"])
            fee = Trades[i, "sellQty"]*(trade["sellType"] == "limit" ? params["MakerFee"] : params["TakerFee"])
            Trades[i, "sellFee"] = fee*Trades[i, "avgSellPrice"]
            Trades[i, "totalFees"] += fee*Trades[i, "avgSellPrice"]
            Trades[i, "completeTime"] = parseTime(req["filled_at"])

            sellValue = Trades[i,"sellQty"]*Trades[i, "avgSellPrice"]*(1-fee)
            Trades[i, "PL_dollars"] = sellValue - trade["value"]
            Trades[i, "PL_pc"] = 100(sellValue/trade["value"] - 1)

            # Update order value and liquidity in case there is a discrepancy between intended and actual trade value
            Account["Value"] += Trades[i, "PL_dollars"]
            Account["Liquidity"] += sellValue

        elseif trade["createdAt"] + params["testPeriod"] < now() # trade too old, kill
            p = getCurrentPrice(trade["coin"])
            query = Dict(
                "limit_price" => p
            )
            HTTP.patch(
                ENV["PAPER_ENDPOINT"]*"/orders/"*trade["sellID"],
                header, query = query
            )
        end
        # Save every loop in case something breaks
        @save "liveVersion/bot1Paper/Account.jld2" Account params
        CSV.write("liveVersion/bot1Paper/tradeLog.csv", Log)
    end
end

# i=6;trade=eachrow(Trades)[i]
# req = HTTP.get(
#     ENV["PAPER_ENDPOINT"]*"/orders/"*ord["orderID"],
#     getHeader
# ).body |> String |> JSON.parse

# Trades[i,"sellID"] = "a5cb4853-eef9-4a93-8779-4081ba91859f"
# req["symbol"]
# Log[16,"coin"] = "CRV"

function main()

    # ========================================
    # =====   Update Ongoing Positions   =====
    # ========================================

    # Update orders
    # - Update info for orders which have execute
    # - Place sell order on orders which have executed
    # - Close orders which are too old
    println("Updating orders...")
    updateOrders()
    println("Done\n")
    
    # Update Trades
    # - Update info for trades which have finished
    # Close trades which are too old
    println("Updating trades...")
    updateTrades()
    println("Done\n")
    
    # ========================================
    # ============   New Trades   ============
    # ========================================

    # Update quote data
    println("Updating quote data...")
    updateQuotes()
    println("Done\n")
    
    # Get trades from algorithm
    println("Getting predictions...")
    preds = bot1()
    println("Done\n")
    
    # Execute trades from algorithm
    println("Placing new orders...")
    # Looping to make sure task is computed syrnchronously
    for pred in preds 
        placeOrder(pred)
    end
    println("Done")
end

main()

# println("Winner")
# sleep(5)