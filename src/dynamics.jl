using DynamicalSystems
import DynamicalSystems: dimension
using LinearAlgebra: mul!

struct NeuralModel{T<:Real,S<:Real}
    adj::Matrix{T}
    spikes::Vector{Vector{S}}
    
    function NeuralModel(adj::Matrix{T}) where T
        @assert size(adj, 1) == size(adj, 2)
        new{T,Int64}(adj, [Vector{Int64}() for _ in 1:size(adj, 1)])
    end
end
function register_spike!(m::NeuralModel{T,S}, spike_idx, t::S) where {T, S}
    push!(m.spikes[spike_idx], t)
end
DynamicalSystems.dimension(m::NeuralModel) = length(m.spikes)

function (m::NeuralModel)(out::AbstractArray{<:Real}, u, p, t)
    N = dimension(m)
    # N = length(m.spikes)
    k, T, reset, rest, Δt = p
    v, s = u[1:N], u[N+1:2N]
    vn = v + Δt * (-k .* (v.-rest) .+ m.adj * s .+ randn(N))
    spikes = vn .> T
    vn = ifelse.(spikes, reset, vn)
    out[1:N] = vn
    out[N+1:2N] = spikes
    return nothing
end

potentials(A::StateSpaceSet) = A[:, 1:size(A, 2) ÷ 2]
function spikes(A::StateSpaceSet)
    N = size(A, 2) ÷ 2
    s = A[:, N+1:2N]
    [findall(>(0), c) for c in eachcol(s)]
end
