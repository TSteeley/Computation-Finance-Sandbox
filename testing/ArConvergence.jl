using LinearAlgebra

X = [1, 2, 3, 4]

d = length(X)

β = [1, 0.4, 0.4, 0.4]

T = vcat(I(d)[:,1]', β', I(d)[:,2:3]')

P = eigvecs(T)

D = round.(inv(P)*T*P, digits = 5)

P*D^100*inv(P)
T^100

λ = eigvals(T)[1]
det(T - λ*I(d))

A = (T - λ*I(d))[[1,collect(3:d)...,2],:]
det(A)