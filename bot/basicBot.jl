using Distributions, Plots, CustomPlots
using JLD2

n = 10
σ = sqrt(1/n)
ξ = Normal(0, σ)
N = 5000

function market(θ::AbstractVector{Float64})
    lb, ub = exp.(θ .* [-1, 1])
    score = 0
    X = cumsum(randn(N, n) .* σ/√n, dims = 2)
    for x in eachrow(X)
        idx = findfirst((x .< lb) .|| (x .> ub))
        score += isnothing(idx) ? x[end] : x[idx]
    end
    return score / N
end

prior = product_distribution(
    Exponential(1),
    Exponential(1)
)

######################
####  Initialise  ####
######################

θ = rand(prior, 500)
S = market.(eachcol(θ))

@load "bot/data/basic_1.jld2" θ S

####################
####  Run sims  ####
####################

i = 1
for _ in 1:100
    idx = sortperm(S, rev = true)

    clamp!(θ, 0.0, 3.0)

    S = S[idx]
    θ = θ[:,idx]
    
    Σ = 2*cov(θ[:,1:250]')
    Y = MvNormal(Σ)
    
    for i in 1:10, j in 251:500
        θp = θ[:,rand(1:250)] + rand(Y)
    
        if all(θp .≥ 0)
            S[j] = market(θp)
            θ[:,j] = θp
        end
    end

    @save "bot/data/basic_1.jld2" θ S
    
    xlim = maximum(abs.(1 .- [minimum(exp.(-θ[1,:])), maximum(exp.(θ[2,:]))]))
    bins = range(0, xlim, length = 50)
    
    p0 = plot()
    histogram!(p0, S, label = "", title = "Score")
    vline!(p0, [0], lc = :red, lw = 2, label = "")
    
    p1 = plot(xlim = [1-1.2xlim, 1+1.2xlim], title = "Iteration $i")
    histogram!(p1, exp.(-θ[1,:]), bins = 1 .- bins[end:-1:1], normalize = :pdf, label = "Stop Loss", fc=:red, lw=0.5)
    histogram!(p1, exp.(θ[2,:]), bins = 1 .+ bins, normalize = :pdf, label = "Take Profit", fc=:green, lw=0.5)
    
    p = plot(p1, p0, layout = [1,1], size = [500, 350])
    display(p)
    i += 1
end