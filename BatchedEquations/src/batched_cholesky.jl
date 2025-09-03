# https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
# https://homepages.inf.ed.ac.uk/imurray2/pub/16choldiff/choldiff.pdf

# non-inPlace versions
function batched_cholesky(A::AbstractMatrix)
    N = size(A,1)
    L = fill!(similar(A),0)
    for i in 1:N, j in 1:i
        if j == i
            L[i,i] = sqrt(A[i,i] - L[i,1:i-1]'*L[j,1:i-1])
        else
            L[i,j] = (A[i,j] - L[i,1:j-1]'*L[j,1:j-1])/L[j,j]
        end
    end
    return L
end

function batched_cholesky(A::AbstractArray{T,3}) where T <: Real
    N = size(A,1)
    L = fill!(similar(A),0)
    L[1,1,:] = sqrt.(A[1,1,:])
    for i in 2:N, j in 1:i
        if j == i
            L[i,i,:] = sqrt.(A[i,i,:] .- sum(L[i,1:i-1,:].*L[i,1:i-1,:];dims=1)[:])
        # elseif j-1 == 0
        #     L[i,j,:] = A[i,j,:] ./ L[j,j,:]
        else
            L[i,j,:] = (A[i,j,:] - sum(L[i,1:j-1,:].*L[j,1:j-1,:];dims=1)[:]) ./ L[j,j,:]
        end
    end
    return L
end

# In-place batched_colesky functions
function level2partition(A::AbstractMatrix,j)
    r = @view A[j,1:j-1]
    d = @view A[j,j]
    B = @view A[j+1:end,1:j-1]
    c = @view A[j+1:end,j]
    return r, d, B, c
end

function level2partition(A::AbstractArray{T,3},j) where T <: Real
    r = @view A[j:j,1:j-1,:]
    d = @view A[j:j,j:j,:]
    B = @view A[j+1:end,1:j-1,:]
    c = @view A[j+1:end,j:j,:]
    return r, d, B, c
end

function batched_cholesky!(A::AbstractMatrix)
    N = size(A,1)
    for j in 1:N
        r, d, B, c = level2partition(A,j)
        d .= sqrt(d .- r'*r)
        c .= (c-B*r)./d
    end
    A.=tril(A)
end

function batched_cholesky!(A::AbstractArray{T,3}) where T <: Real
    N = size(A,1)
    for j in 1:N
        r, d, B, c = level2partition(A,j)
        d .= sqrt.(d .- r⊠batched_transpose(r))
        c .= (c-B ⊠ batched_transpose(r))./batched_transpose(d)
    end
    A .= mapslices(tril,A, dims=(1,2))
end

# Differentitation rules

function ChainRulesCore.rrule(::typeof(batched_cholesky),A::AbstractMatrix)
    L = batched_cholesky(A)
    N = size(A,1)
    function dL(Δ)
        dA = unthunk(Δ) |> copy
        for j = N:-1:1
            r, d, B, c = level2partition(L, j)
            r̄, d̄, B̄, c̄ = level2partition(dA, j)
            d̄ .-= c'*c̄ ./ d
            d̄ ./= d
            c̄ ./= d
            r̄ .-= (hcat(d̄[:], c̄[:]')*vcat(r[:,:]',B[:,:]))[:]
            B̄ .-= c̄*r'
            d̄ ./= T(2)
        end
        dA .= tril(dA)
        return dA
    end
    function batched_chollesky_pullback(Δ)
        dΔ = @thunk(dL(Δ))

        return NoTangent(), dΔ
    end
    return L, batched_chollesky_pullback
end

function ChainRulesCore.rrule(::typeof(batched_cholesky), A::AbstractArray{T,3}) where T <: Real
    L = batched_cholesky(A)
    N = size(A,1)
    function dL(Δ)
        dA = unthunk(Δ) |> copy
        for j = N:-1:1
            r, d, B, c = level2partition(L, j)
            r̄, d̄, B̄, c̄ = level2partition(dA, j)
            d̄ .-= batched_transpose(c)⊠c̄ ./ d
            d̄ ./= d
            c̄ ./= d
            r̄ .-= hcat(d̄, batched_transpose(c̄)) ⊠ vcat(r,B)
            B̄ .-= c̄ ⊠ r
            d̄ ./= T(2)
        end
        @views for dA in eachslice(dA, dims=3)
            dA .= tril(dA)
        end
        return dA
    end
    function batched_chollesky_pullback(Δ)
        dΔ = @thunk(dL(Δ))

        return NoTangent(), dΔ
    end
    return L, batched_chollesky_pullback
end

# TODO: Implement maybe?
# function level3partition(A,j,k,N)
#     R = @view A[j:k,1:j-1]
#     D = @view A[j:k,j:k]
#     B = @view A[k+1:N,1:j-1]
#     C = @view A[k+1:N,j:k]
#     return R, D, B, C
# end

# function chol_blocked(A, Nb)
#     N = size(A,1)
#     for j in 1:Nb:N
#         k = min(N, j+Nb-1)
#         R, D, B, C = level3partition(A,j,k,N)
#         D .-= tril(R*R')
#         chol!(D)
#         C .-= B*R'
#         C .= C*(tril(D)\I)'
#     end
#     tril(A)
# end


# function ChainRulesCore.rrule(::typeof(batched_cholesky!),A::AbstractArray{T,2}) where T <: Real
#     L = batched_cholesky(A)
#     function batched_cholesky!_pullback(Δ)
#         N = size(A,1)
#         for j = N:-1:1
#             r, d, B, c = level2partition(L, j)
#             r̄, d̄, B̄, c̄ = level2partition(Δ, j)
#             d̄ .-= c'*c̄ ./ d
#             d̄ ./= d
#             c̄ ./= d
#             r̄ .-= (hcat(d̄[:], c̄[:]')*vcat(r[:,:]',B[:,:]))[:]
#             B̄ .-= c̄*r'
#             d̄ ./= T(2)
#         end
#         Δ .= tril(Δ)

#         return NoTangent, Δ
#     end
#     return L, batched_cholesky!_pullback
# end

