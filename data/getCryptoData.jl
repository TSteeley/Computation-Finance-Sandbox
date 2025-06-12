using FinData, Dates, ProgressBars, Base.Threads
using TOML

coins = ["AAVE", "AVAX", "BAT", "BCH", "BTC", "CRV", "DOGE", "ETH", "GRT", "LINK", "LTC", "MKR", "PEPE",  "SHIB","SOL", "SUSHI", "TRUMP", "UNI", "XRP", "XTZ", "YFI"] # , "DOT"
coinQuotes = [k for k in keys(TOML.parsefile("data/cryptoQuotes/data.toml"))]
coinBars = [k for k in keys(TOML.parsefile("data/cryptoBars/data.toml"))]
for coin in coins |> ProgressBar
    println(coin)
    if coin ∉ coinQuotes
        gatherCryptoQuotes(coin, startTime="2023-01-01", endTime=string(now()))
    end
    if coin ∉ coinBars
        gatherCryptoBars(coin, startTime="2023-01-01", endTime=string(now()))
    end
end

# for coin in coins
#     if coin ∉ coinQuotes
#         println(coin)
#     end
#     # if coin ∉ coinBars
#     #     println(coin)
#     # end
# end




