using HTTP, JSON, DataFrames, TOML
using Dates, TimeZones, JLD2, DotEnv
DotEnv.load!()
# Head

header = Dict(
    "accept" => "application/json",
    "APCA-API-KEY-ID" => ENV["PAPER_KEY"],
    "APCA-API-SECRET-KEY" => ENV["PAPER_SECRET_KEY"],
)

HTTP.get(
    ENV["PAPER_ENDPOINT"],
    headers,
    query=query,
)

# ============================================================
# ========================   Account   =======================
# ============================================================

resp = HTTP.get(
    ENV["PAPER_ENDPOINT"]*"/account",
    header,
).body |> String |> JSON.parse



# ============================================================
# ========================   Orders   ========================
# ============================================================

resp = HTTP.get(
    ENV["PAPER_ENDPOINT"]*"/orders",
    header,
    query=query,
).body |> String |> JSON.parse

# Create Order
query = Dict(
    "symbol" => "",
    "notional" => "",
    "side" => "buy",
    "type" => "limit",
    "time_in_force" => "gtc",
    "limit_price" => 1, # Only if limit order
    "order_class" => "simple",
    "position_intent" => "buy_to_open",
)

# "type" => "market", "limit", "stop", "stop_limit", "trailing_stop"

# Cancel Order

# Get All Orders
query = Dict(
    "status" => "open",
    "limit" => 500,
    "after" => "2025-01-01",
)

req = HTTP.get(
    
)

# ============================================================
# =======================   Positions   ======================
# ============================================================

