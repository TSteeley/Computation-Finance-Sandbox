using FinData

BTCdata = fetchCryptoData("BTC")

dW(σ) = rand(Normal(0, σ))

function model(a, b, x0, σ)
    p = zeros(size(BTCdata, 1))
    p[1] = x0
    for i in 2:size(BTCdata,1)
        p[i] = p[i-1] + dW(σ) + 2a*(x0 + b*i - p[i-1])
    end
    return p
end

prior = product_distribution(
    Exponential(0.1), # a
    Normal(0,10), # b
    Normal(BTCdata[1,"vw"], 500000), # x0
    Exponential(10), # σ
)

ρ(p) = norm(BTCdata[!,"vw"] - p)

n = 500_000
p = rand(prior, n)
ρn = zeros(n)
for i in 1:n
    ρn[i] = ρ(model(p[:,i]...))
end

idx = sortperm(ρn)

X = 0:296
P = plot()
squarify!(P, X, BTCdata[!,"vw"], lc = :black, label = "True", ylims = (82000, 85000))

for _ in 1:100
    r = rand(idx[1:50])
    squarify!(P, X, model(p[:,r]...), lc = :blue, la = 0.1, label = "")
end

P