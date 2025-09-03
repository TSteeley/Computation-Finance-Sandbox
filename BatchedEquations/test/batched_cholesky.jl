using Test, ChainRulesTestUtils
using LinearAlgebra, MLUtils
using FiniteDifferences
using CUDA


# Tests are performed with n m×m randomly sampled pos-def matrices
m = 5 # dimensionality
n = 100 # number of matrices
# fdm has to do very small perturbations or else it goes out of domain
fdm = FiniteDifferenceMethod([-10:10;], 1, max_range=0.0001)

# ============================================================
# =================   Test a single matrix   =================
# ============================================================
# Random test value
A = randn(m,m) |> x -> x * x' + I(m)

# Show solutions are the same as that of LinearAlgebra.cholesky
@test batched_cholesky(A) ≈ cholesky(A).L
L = copy(A) # to test in-place version
batched_cholesky!(L)
@test L ≈ cholesky(A).L

# Test differentiation
test_rrule(batched_cholesky, A, fdm=fdm)

# ============================================================
# =================   For a set of matrices   ================
# ============================================================
# Random test values
B = randn(m,m,n) |> x -> x ⊠ x' .+ I(m)

# Show solutions are the same as that of LinearAlgebra.cholesky
@test batched_cholesky(B) ≈ mapslices(b -> cholesky(b).L,B,dims=(1,2))
L = copy(B) # to test in-place version
batched_cholesky!(L)
@test L ≈ mapslices(b -> cholesky(b).L,B,dims=(1,2))

# Test differentiation
test_rrule(batched_cholesky, B, fdm=fdm)

# ============================================================
# ==================   Test CUDA support   ===================
# ============================================================

# Show solutions are the same as that of LinearAlgebra.cholesky
# @test cpu(batched_cholesky(cu(B))) ≈ mapslices(b -> cholesky(b).L,B,dims=(1,2))
# L = copy(B) |> cu # to test in-place version
# batched_cholesky!(L)
# @test cpu(L) ≈ mapslices(b -> cholesky(b).L,B,dims=(1,2))

# Test differentiation
# test_rrule(batched_cholesky, cu(B), fdm=fdm)
