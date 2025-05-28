#=
    Optimisation.jl

    Author: Thomas P. Steele

    Goal: Given some function f(x) what is the probability x maximises/ minimises f(x). 
    I am thinking instead find the probability π( f'(x) = 0 | x )
=#

using Distributions, Plots, LinearAlgebra

σ = 1
ε() = rand(Normal(0,σ))

# main function 
f(x) = x^2 + ε()
df(x) = 2x + ε()

# Obv f(x) is not differentiable, however, work with me here.

prior = Normal(0, 5)

n = 10_000_000
p = rand(prior, n)
ρn = zeros(n)

for i in 1:n
    ρn[i] = norm(df(p[i]))
end

idx = sortperm(ρn)

histogram(p[idx[1:1000]], bins = 100, normalize = :pdf)

scatter(p[idx[1:1000]], f.(p[idx[1:1000]]))
scatter(p, f.(p))

