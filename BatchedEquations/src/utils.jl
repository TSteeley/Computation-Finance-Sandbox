

# Outer product and scale
function OPAS(x::AbstractArray)
    x ⊠ batched_transpose(x) / size(x,2)
end
function OPAS(x::AbstractMatrix)
    x * x' / size(x,2)
end