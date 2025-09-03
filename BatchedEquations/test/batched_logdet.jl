

m=5
n=100
A = randn(m,m,n) |> x -> x ⊠ batched_transpose(x) .+ I(m);

@test [logdet(a) for a in eachslice(A,dims=3)] ≈ batched_logdet(A)


A = randn(5,5) |> x -> x*x' + I(5)
B = randn(5,5,2) |> x -> x ⊠ batched_transpose(x) .+ I(5)

fdm = FiniteDifferenceMethod([-10:10;], 1, max_range=0.001)

test_rrule(batched_logdet, A, fdm=fdm)
test_rrule(batched_logdet, B, fdm=fdm)
test_rrule(batched_logdet, reshape(A,5,5,1), fdm=fdm)

A = A+I(5)
dA = jacobian(fdm, batched_logdet, B)[1]
reshape(dA, 5, 5)

ldA, dldA = Flux.pullback(batched_logdet, A)
dldA(1)[1]

ldB, dldB = Flux.pullback(batched_logdet, reshape(A,5,5,1))
dldB(1)[1]
