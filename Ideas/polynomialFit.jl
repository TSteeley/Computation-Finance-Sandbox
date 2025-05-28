using Plots, CustomPlots, LinearAlgebra
using Combinatorics

f(x) = (1+25x^2)^-1

D = [-2, 2]

k = 61
X = D[1] .+ (1 .- cos.(π*range(0,1,length=k))).*diff(D)/2
Y = f.(X)

ℓ(x,j) = prod([(x - X[m])/(X[j]-X[m]) for m in 1:k if m .!= j])
L(x) = ℓ.(x, 1:k)' * Y


aᵢ = j -> [-X[m] for m in 1:k if m .!= j]
function findCoeffs(X, k)
    β = Matrix{Float64}(undef, k, k)
    β[:, k] .= 1
    for (j, a) in enumerate(aᵢ.(1:k))
        β[j, 1] = prod(a)
        for i in 1:k-2, 
            β[j, k-i] = sum(c -> prod(a[j] for j in c), combinations(1:k-1, i))
        end
        β[j,:] ./= prod(a .+ X[j])
    end
    return β
end

β = findCoeffs(X, k)

L2(x) = (β * x.^[0:k-1;])' * Y

fplot(D, f, label = "Original")
scatter!(X, Y, mc=:black, label = "")
fplot!([-2, 2], L, label = "")
fplot!([-2, 2], L2, label = "")

@benchmark findCoeffs(X, k)

# ============================================================
# ====================   Implementations   ===================
# ============================================================

# Recursive
function e(X::Vector, k::Int)
    if k == 0
        return 1.0
    elseif k == 1
        return sum(X)
    else
        n = length(X)
        return sum(j -> X[j]*e(X[j+1:end], k-1), 1:n-k+1)
    end
end

β2 = vcat([[e(aᵢ(j), k-m-1) for m in 0:k-1]' / prod(aᵢ(j) .+ X[j]) for j in 1:k]...)

findall(round.(β - β2, digits = 2) .!= 0)


# Big fuck off one liner
β = hcat(vcat([reverse([sum(c -> prod(a[j] for j in c), combinations(1:k-1, i)) for i in 1:k-1])' ./ prod(aᵢ(j) .+ X[j]) for (j, a) in enumerate(aᵢ.(1:k))]...), [1/prod(aᵢ(j) .+ X[j]) for j in 1:k])