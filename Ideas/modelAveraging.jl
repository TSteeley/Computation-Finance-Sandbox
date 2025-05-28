using Distributions, StatsBase
using Plots, CustomPlots
using LinearAlgebra

import Distributions: Normal
Normal(μ::Vector,σ::Number) = Normal(μ[1], σ)
Normal(μ::Matrix,σ::Number) = Normal(μ[1], σ)

(F::Vector{Function})(x) = [f(x) for f in F]

# ============================================================
# =========================   Data   =========================
# ============================================================

N = 101 # Data points
ϵ = rand(Normal(0, 1), N) # Noise
X = range(0, 20, length = N)
Y = @. 20sqrt(1-(X/20-1)^2) + ϵ

scatter(X, Y, aspect_ratio = :equal)

# ============================================================
# ====================   Constant Model   ====================
# ============================================================
# Y ∼ μ + ε

function constant(X::AbstractVector{Float64}, Y::Vector{Float64})
    μ = mean(Y)
    σ = std(Y)
    return θ -> Normal(μ, σ)
end

M0 = constant(X, Y)

p0 = plot(title = "Constant Model", legend_position = :bottomright)
scatter!(p0, X, Y, label = "Data")
fplot!(p0, X, x -> mean(M0(x)), ribbon = std.(M0.(X)), label = "ConstantModel")

# ============================================================
# =====================   Linear Model   =====================
# ============================================================

function linear(X::AbstractVector{Float64}, Y::Vector{Float64})
    X = hcat(repeat([1], length(X)), X)

    β = inv(X'*X)*X'*Y
    σ = std(Y - X*β)
    return x -> Normal(β[1] + β[2]*x, σ)
end

M1 = linear(X, Y)

p1 = plot(title = "Linear Model", legend_position = :bottomright)
scatter!(p1, X, Y, label = "Data")
fplot!(p1, X, x -> mean(M1(x)), ribbon = std.(M1.(X)), label = "Model")

# ============================================================
# =======================   Order N   ========================
# ============================================================

function OrderN(X::AbstractVector{Float64}, Y::Vector{Float64}; order = 2)
    X = hcat(ones(length(X)),[X.^i for i in 1:order]...)
    
    β = inv(X'*X)*X'*Y
    σ = std(Y - X*β)
    
    return x -> Normal(hcat(1,x.^[1:order;]')*β, σ)
end

order = 10
M2 = OrderN(X, Y, order = order)

p2 = plot(title = "Order $order", legend_position = :bottomright)
scatter!(p2, X, Y, label = "Data")
fplot!(p2, X, x -> mean(M2(x)), ribbon = std.(M1.(X)), label = "Model")

pn = plot(title = "Models", legend_position = :bottomright)
scatter!(pn, X, Y, label = "Data")
for i in 0:5
    M = OrderN(X, Y, order = i)
    fplot!(pn, X, x -> mean(M(x)), ribbon = std.(M1.(X)), label = "Order $i")
end
display(pn)

# ============================================================
# ================   Bayesian Model Average   ================
# ============================================================

n = 8
M = Vector{Function}(undef, n+1)
w = Vector{Float64}(undef, n+1)
for i in 0:n
    M[i+1] = OrderN(X, Y, order = i)
    w[i+1] = exp.(sum(logpdf.(M[i+1].(X), Y)) - (i+2).*log(N)/2)
end

BAMμ(x::Number) = mean(mean.(M(x)), weights(w))
BAMσ(x::Number) = mean(var.(M(x)), weights(w)) |> sqrt

p = plot(title = "Bayesian Model Average", legend_position = :bottomright)
scatter!(p, X, Y, label = "Data")
fplot!(p, X, BAMμ, ribbon = BAMσ, label = "Model")


fplot((0, 30), BAMμ, ribbon = BAMσ, label = "")
fplot!((0, 30), x -> 20sqrt(1-(x/20-1)^2))

