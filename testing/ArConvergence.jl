using Distributions, StatsBase, ProgressBars
using FinData, Dates, DataFrames
using LinearAlgebra, Base.Threads, JLD2
using Plots, CustomPlots
# using Flux
include("../functions.jl")

function Format_TS_data(TS::AbstractVector{T}, P::Int, D::Int) where T <: Real
    if D != 0
        for _ in 1:D
            TS = diff(TS)
        end
    end

    X = hcat(
        repeat([1], length(TS)-P), 
        [TS[p:end-P+p-1] for p in P:-1:1]...
    )
    Y = TS[P+1:end]
    return X, Y
end

function BayesPost(X, Y; N::Int=50_000)
    n, m = size(X)

    # Priors
    β₀ = ones(m)*0
    λ = I(m)*0.5
    a = 0
    u = 0

    # Sample Posterior Distributions
    pΛ = Gamma(a + n/2, inv(u+0.5*(Y'*Y + β₀'*λ*β₀ - (X'*Y+λ*β₀)'*inv(X'*X+λ)*(X'*Y+λ*β₀))))
    Λ = rand(pΛ, N)
    β = inv(X'*X+λ)*(X'*Y+λ*β₀) .+ sqrt(inv(X'*X+λ))*randn(m, N) ./ sqrt.(Λ')

    # Posterior expectations
    μ = mean(β[1,:] ./ (1 .- sum(β[2:end,:]', dims = 2)[:])) # Mean value
    V = Λ .* (1 .- sum(β[2:end,:]', dims = 2).^2)[:] |> V -> mean((V[V .> 0]).^(-0.5)) # Variance
    return μ, V
end

tStep = Minute(5) # Chunking of quotes
step = Hour(6) # How often esimates are made on data
MW = Day(3) # Maximum wait for an order to execute
trainPeriod = Week(4) # Window size
testPeriod = Week(2) # Time a trade has to completely finish
P = 6 # How long back AR looks

coin = "BTC"
data = loadCryptoQuotes(coin)
filter!(d -> d.bp != 0, data)
transform!(data, :t => ByRow(t -> round(t, tStep, RoundDown)) => :t)

minData = combine(groupby(data, :t),
    :bp => last => :bp,
    # :ap => mean => :ap,
)
fillMissing!(minData, step = tStep)
transform!(minData, :bp => ByRow(log) => :X)

T0 = round(minData.t[1], Day, RoundDown)+trainPeriod+step
MaxT = minData.t[end]-testPeriod

T = range(T0, MaxT, step=step) |> rand
train = @view minData[T-trainPeriod .≤ minData.t .< T,:]

X, Y = Format_TS_data(train.X, P, 0)
n, m = size(X)

# N = 50_000

# # Priors
β₀ = ones(m)*0
λ = I(m)*0.01
a = 0
u = 0

N = 50_000
# Sample Posterior Distributions
pΛ = Gamma(a + n/2, inv(u+0.5*(Y'*Y + β₀'*λ*β₀ - (X'*Y+λ*β₀)'*inv(X'*X+λ)*(X'*Y+λ*β₀))))
Λ = rand(pΛ, N)
β = inv(X'*X+λ)*(X'*Y+λ*β₀) .+ sqrt(inv(X'*X+λ))*randn(m, N) ./ sqrt.(Λ')

[std(Y-X*b) for b in eachcol(β)] |> mean
sqrt(1/mode(pΛ))

b = (X'*X)\X'*Y
std(Y - X*b)

# # Posterior expectations
μ = mean(β[1,:] ./ (1 .- sum(β[2:end,:]', dims = 2)[:])) # Mean value
# V = Λ .* (1 .- sum(β[2:end,:]', dims = 2).^2)[:] |> V -> mean((V[V .> 0]).^(-0.5)) # Variance
V = Λ .* (1 .- β[2:end]'*β[2:end]) |> V -> mean((V[V .> 0]).^(-0.5)) # Variance

N = 10_000_000
y = zeros(N)
x = zeros(N, P+1)
β = inv(X'*X+λ)*(X'*Y+λ*β₀)
μ = β[1]/(1-sum(β[2:end]))
σ = std(Y - X*β)
var(Y - X*β)
yn = [1 ; ones(P)*μ]

A = zeros(P+1,P+1)
A[1,1] = 1
A[2,:] .= β
A[3:end,2:end-1] += I(P-1)

# Burn in
for _ in 1:100
    yn = A*yn
    yn[2] += σ*randn()
end

for i in 1:N
    x[i,:] .= yn
    yn = A*yn
    yn[2] += σ*randn()
    y[i] = yn[2]
end

σ
std(y - x*β)

mean(y)
β[1]/(1-ones(P)'*β[2:end])

var(y)
std(y)

ϕ = β[2:end]


var(y)
((y .- μ)'*(y .- μ))/(N+2)
(σ^2)/(1-β[2:end]'*β[2:end])

histogram(y[1:5:end], label = "", normalize=:pdf)
vline!([μ ± 1.96std(y)], label="95CI")
vline!([quantile(y, [0.025, 0.975])], label="c95cred int")
vline!([μ ± 1.96sqrt(γ[1])])

(quantile(y, [0.025, 0.975]) .- μ)/1.96 |> x -> mean(abs.(x))
(quantile(y, [0.15865, 0.84134]) .- μ) |> x -> mean(abs.(x))



y
x = hcat(ones(length(y)-P), [y[P-k+1:end-k] for k in 1:P]...)
yp = y[P+1:end]
x2 = x[:,2]
x3 = x[:,3]

β
b = inv(x'*x)*x'*yp

x'*x / N

cov(x2,x3)

σ^2
var(yp - x*b)/(1-b[2:end]'*b[2:end])

var(y)

x'*x / N



mean(x2.^2)
mean(x2.*x3)
μ.^2

cov(x2,x3)
(x2 .- mean(x2))'*(x3 .- mean(x3))/(N-1)
(x2 .- μ)'*(x3 .- μ)/N

x2'*x3/N - mean(x2)*mean(x3)
x2'*x3/N - μ^2

cov(y[3:end], x2)

y[3:end]'*x2/N-mean(x2)^2
y[3:end]'*x2/N-μ^2

μ2 = [1 ; ones(P)μ]

β'*(μ2*μ2')*β - μ^2

function arVar(ϕ::Vector{<:Real}, V::Float64)
    
end

# Direct inversion
ϕ = β[2:end]
b = [σ^2 ; zeros(P)]
T = zeros(P+1,P+1)
T[1,1] = 1
T[1,2:end] = -ϕ

for k in 1:P
    T[k+1, k+1] = 1
    for j in 1:P
        idx = abs(k-j)+1
        T[k+1, idx] -= ϕ[j]
    end
end

for i in 1:P
    T[i,i] = 1
end

γ = T \ b
var(y)
sqrt(γ[1])

@benchmark T \ b # 551.326 ns ± 79.520 ns
@benchmark inv(T)*b # 1.288 μs ±   9.955 μs

# Iterative
ϕ = β[2:end]
Γ = zeros(P+1,P+1)
# γ = γ[2:end]
γ = -log.(rand(P+1))
γ0 = -log(rand())

for _ in 1:100
    γold = copy(γ)

    for i in 1:P+1, j in 1:P+1
        Γ[i,j] = i == j ? γ0 : γ[abs(i-j)]
    end

    γ = Γ*β # ϕ
    norm(γ - γold) > 1e-10 || break

    γ0 = γ'*ϕ + σ^2
    norm(γ0 - γ0old) > 1e-10 || break
end

Γ*β
γ0

inv(Γ)*γ

γ'*ϕ + σ^2 |> sqrt
std(y)

ϕ' * γ

for i in 1:P+1, j in 1:P+1
    T[i,j] = i == j ? 1 : ϕ[abs(i-j)]
end

inv(A)*b


# Fuck it, just doing by MC


function arPost(X, Y; N::Int=50_000)
    # Priors
    β₀ = ones(m)*0
    λ = I(m)*0.1
    a = 0
    u = 0
    
    pΛ = Gamma(a + n/2, inv(u+0.5*(Y'*Y + β₀'*λ*β₀ - (X'*Y+λ*β₀)'*inv(X'*X+λ)*(X'*Y+λ*β₀))))
    
    mβ = inv(X'*X+λ)*(X'*Y+λ*β₀)
    σβ = sqrt(inv(X'*X+λ))
    
    y = zeros(N)
    yn = [1 ; ones(P)*mβ[1]/(1-sum(mβ[2:end]))]
    
    # Transition matrix
    A = zeros(P+1,P+1)
    A[1,1] = 1
    A[3:end,2:end-1] += I(P-1)
    
    # Burn in
    for _ in 1:100
        σ = 1/sqrt(rand(pΛ))
        β = mβ + σβ*randn(m)*σ
        A[2,:] .= β
        yn = A*yn
        yn[2] += σ*randn()
    end
    
    for i in 1:N
        σ = 1/sqrt(rand(pΛ))
        β = mβ + σβ*randn(m)*σ
        A[2,:] .= β
        yn = A*yn
        yn[2] += σ*randn()
        y[i] = yn[2]
    end
    return mean(y), std(y)
end

mean(y)
std(y)

β = inv(X'*X+λ)*(X'*Y+λ*β₀)
μ = β[1]/(1-sum(β[2:end]))
std(Y - X*β)
var(Y - X*β)