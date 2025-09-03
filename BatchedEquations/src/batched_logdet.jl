

function batched_logdet(A::AbstractMatrix{T}) where T <: Real
    L = batched_cholesky(A)
    return 2sum(log, diag(L))
end
function batched_logdet(A::AbstractArray{T,3}) where T <: Real
    m, _, n = size(A)
    L = batched_cholesky(A)
    return 2sum(log, L[(1:m+1:m^2) .+ (0:n-1)'*m^2], dims=1)[:]
end

# function ChainRulesCore.rrule(::typeof(batched_logdet), A::AbstractArray{T,3}) where T <: Real
#     V = batched_logdet(A)
#     m = size(A,1)
#     function batched_logdet_pullback(Δ)
#         Δ = unthunk(Δ)
#         dA = copy(A)
#         _I = fill!(similar(A,m,m), 0)
#         _I[1:m+1:m^2] .= 1
#         @views for dA in eachslice(dA, dims=3)
#             dA .= tril(dA \ _I)
#             dA[[i for i in 2:m^2 if i ∉ 1:m+1:m^2]] .*= 2
#         end
#         return NoTangent(), Δ .* dA
#     end
#     return V, batched_logdet_pullback
#     # pullback = (NoTangent(), ∂batched_logdet(A))
#     # return V, pullback
# end

# function ChainRulesCore.rrule(::typeof(batched_logdet), A::AbstractMatrix{T}) where T <: Real
#     V = batched_logdet(A)
#     function batched_logdet_pullback(Δ)
#         Δ = unthunk(Δ)
#         return NoTangent(), Δ .* tril(A \ I) .* (2ones(5,5)-I(5))
#     end
#     return V, batched_logdet_pullback
# end


# function ∂batched_logdet(A::AbstractMatrix{T}) where T <: Real
#     function ∂(Δ)
#         Δ = unthunk(Δ)
#         dA = tril(A \ I)
#         dA[[i for i in 2:m^2 if i ∉ 1:m+1:m^2]] .*= 2
#         return Δ .* dA
#     end
#     return ∂
# end
# function ∂batched_logdet(A::AbstractArray{T,3}) where T <: Real
#     m = size(A,1)
#     _I = fill!(similar(A,m,m), 0)
#     _I[1:m+1:m^2] .= 1
#     # Constructing the identity this dumb way because I cannot figure out
#     # how else to put it onto the gpu dynamically
#     function ∂(Δ)
#         Δ = unthunk(Δ)
#         dA = copy(A)
#         @views for dA in eachslice(dA, dims=3)
#             dA .= tril(dA \ _I)
#             dA[[i for i in 2:m^2 if i ∉ 1:m+1:m^2]] .*= 2
#         end
#         return Δ .* dA
#     end
#     return ∂
# end

# function ChainRulesCore.rrule(::typeof(batched_logdet), A::AbstractArray{T}) where T <: Real
#     V = batched_logdet(A)
#     pullback = (NoTangent(), ∂batched_logdet(A))
#     return V, pullback
# end




