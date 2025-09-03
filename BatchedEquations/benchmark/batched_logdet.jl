
@benchmark batched_logdet($A)
# BenchmarkTools.Trial: 10000 samples with 1 evaluation per sample.
#  Range (min … max):  15.700 μs …  31.246 ms  ┊ GC (min … max):  0.00% … 99.76%
#  Time  (median):     43.800 μs               ┊ GC (median):     0.00%
#  Time  (mean ± σ):   53.550 μs ± 489.922 μs  ┊ GC (mean ± σ):  21.75% ±  2.81%
#  Memory estimate: 152.94 KiB, allocs estimate: 254.

@benchmark [logdet(a) for a in eachslice(A,dims=3)]
# BenchmarkTools.Trial: 10000 samples with 1 evaluation per sample.
#  Range (min … max):  21.900 μs …  22.539 ms  ┊ GC (min … max): 0.00% … 99.76%
#  Time  (median):     30.900 μs               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   34.161 μs ± 225.344 μs  ┊ GC (mean ± σ):  6.58% ±  1.00%
#  Memory estimate: 36.94 KiB, allocs estimate: 406.

