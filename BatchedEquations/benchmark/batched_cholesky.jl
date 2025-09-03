using BenchmarkTools

m = 5
n = 1000
A = randn(m,m) |> x -> x*x'/sqrt(m);
B = randn(m,m,n) |> x -> x ⊠ batched_transpose(x) / sqrt(m);
C = randn(m,m) |> x -> x * x' / sqrt(m) |> cu;
D = randn(m,m,n) |> x -> x ⊠ batched_transpose(x) / sqrt(m) |> cu;

# Benchmark single matrix
@benchmark batched_cholesky!(cA) setup=(cA=copy(A)) evals=1 seconds=3
# BenchmarkTools.Trial: 10000 samples with 1 evaluation per sample.
#  Range (min … max):  300.000 ns …  48.000 μs  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     400.000 ns               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   465.460 ns ± 739.326 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%
#  Memory estimate: 976 bytes, allocs estimate: 20.

@benchmark batched_cholesky(cA) setup=(cA=copy(A)) evals=1 seconds=3
# BenchmarkTools.Trial: 10000 samples with 1 evaluation per sample.
#  Range (min … max):  300.000 ns …  32.600 μs  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     500.000 ns               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   518.600 ns ± 766.171 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%
# Memory estimate: 2.05 KiB, allocs estimate: 52.

@benchmark cholesky(cA).L setup=(cA=copy(A)) evals=1 seconds=3
# BenchmarkTools.Trial: 10000 samples with 1 evaluation per sample.
#  Range (min … max):  100.000 ns …  50.200 μs  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     200.000 ns               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   233.160 ns ± 642.963 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%
#  Memory estimate: 560 bytes, allocs estimate: 5.


# Many matrices
@benchmark batched_cholesky!(cB) setup=(cB=copy(B)) evals=1 seconds=3
# BenchmarkTools.Trial: 2397 samples with 1 evaluation per sample.
#  Range (min … max):  949.300 μs …   3.977 ms  ┊ GC (min … max): 0.00% … 63.01%
#  Time  (median):       1.048 ms               ┊ GC (median):    0.00%
#  Time  (mean ± σ):     1.193 ms ± 458.556 μs  ┊ GC (mean ± σ):  7.96% ± 13.50%
# Memory estimate: 1.02 MiB, allocs estimate: 22539.

@benchmark batched_cholesky(cB) setup=(cB=copy(B)) evals=1 seconds=3
# BenchmarkTools.Trial: 10000 samples with 1 evaluation per sample.
#  Range (min … max):  110.800 μs …  10.846 ms  ┊ GC (min … max):  0.00% … 95.93%
#  Time  (median):     154.800 μs               ┊ GC (median):     0.00%
#  Time  (mean ± σ):   235.744 μs ± 316.002 μs  ┊ GC (mean ± σ):  20.26% ± 16.16%
#  Memory estimate: 1.25 MiB, allocs estimate: 339.

@benchmark mapslices(x -> cholesky(x).L, cB, dims=(1,2)) setup=(cB=copy(B)) evals=1 seconds=3
# BenchmarkTools.Trial: 3406 samples with 1 evaluation per sample.
#  Range (min … max):  596.700 μs …  16.482 ms  ┊ GC (min … max):  0.00% … 95.49%
#  Time  (median):     684.450 μs               ┊ GC (median):     0.00%
#  Time  (mean ± σ):   822.772 μs ± 677.813 μs  ┊ GC (mean ± σ):  12.22% ± 13.05%
#  Memory estimate: 1.10 MiB, allocs estimate: 25468.

# CuArray's

CUDA.@sync @benchmark batched_cholesky!(cC) setup=(cC=copy(C)) evals=1 seconds=3
# BenchmarkTools.Trial: 10000 samples with 1 evaluation per sample.
#  Range (min … max):  533.500 μs … 111.224 ms  ┊ GC (min … max): 0.00% … 32.21%
#  Time  (median):     695.200 μs               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   825.642 μs ±   1.872 ms  ┊ GC (mean ± σ):  1.22% ±  0.54%
#  Memory estimate: 49.72 KiB, allocs estimate: 2085.

# CUDA.@sync @benchmark batched_cholesky(cC) setup=(cC=copy(C)) evals=1 seconds=3
# BenchmarkTools.Trial: 10000 samples with 1 evaluation per sample.
#  Range (min … max):  300.000 ns …  32.600 μs  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     500.000 ns               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   518.600 ns ± 766.171 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%
# Memory estimate: 2.05 KiB, allocs estimate: 52.


# Many matrices
CUDA.@sync @benchmark batched_cholesky!(cD) setup=(cD=copy(D)) evals=1 seconds=3
#  Range (min … max):  44.860 ms … 109.277 ms  ┊ GC (min … max): 0.00% … 23.93%
#  Time  (median):     53.474 ms               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   56.686 ms ±  10.468 ms  ┊ GC (mean ± σ):  0.87% ±  3.29%
#  Memory estimate: 7.56 MiB, allocs estimate: 275677.

CUDA.@sync @benchmark CUDA.@sync batched_cholesky(cD) setup=(cD=copy(D)) evals=1 seconds=3
# @benchmark CUDA.@sync batched_cholesky($D)
# BenchmarkTools.Trial: 1577 samples with 1 evaluation per sample.
#  Range (min … max):  1.200 ms … 111.590 ms  ┊ GC (min … max): 0.00% … 33.71%
#  Time  (median):     1.682 ms               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   1.880 ms ±   2.852 ms  ┊ GC (mean ± σ):  1.27% ±  0.85%
#  Memory estimate: 258.84 KiB, allocs estimate: 9399.