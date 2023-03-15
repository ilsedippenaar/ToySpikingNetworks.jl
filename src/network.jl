using Distributions: MvNormal, Exponential, LogitNormal, Gamma, MixtureModel, sample
using LinearAlgebra: I, norm, Symmetric, diagind
using Distances: Euclidean, pairwise
using Random: AbstractRNG, default_rng
using GeometryBasics: AbstractPoint
using DynamicalSystems
import DynamicalSystems: dimension

export Network

struct Network
    W::Matrix
    pos::Vector{<:AbstractPoint}

    function Network(W, pos)
        @assert size(W, 1) == size(W, 2)
        @assert size(W, 1) == length(pos)
        new(W, pos)
    end
end
dimension(m::Network) = length(m.pos)

circle(n::Integer) = n == 1 ? [0.0, 0.0] : [cos.((0:n-1)*2π/n)'; sin.((0:n-1)*2π/n)']
function circle_of_circles(N::Integer, L::Integer, s::Real)
    nuclei = circle(L) * s
    hcat((circle(N ÷ L + (x ≤ N % L)) .+ nuclei[:, x] for x=1:L)...)
end

function gaussian_blobs(N::Integer, L::Integer, s::Real; dim = 2)
    nuclei = rand(dim, L) .* s
    rand(
        MixtureModel(MvNormal, tuple.(eachcol(nuclei), Ref(I))),
        N,
    )
end

function waxman(positions, α, β)
    W = β .* exp.(-pairwise(Euclidean(), positions)./α)
    W[diagind(W)] .= 0  # no self connections
    W
end
function dalesprinciple!(rng::AbstractRNG, W::Matrix, 𝓲::Real)
    @assert 0 <= 𝓲 <= 1
    W[:, sample(rng, 1:size(W, 2), round(Int, size(W, 2) * 𝓲), replace=false)] *= -1
    W
end

function (m::Network)(out::AbstractArray{<:Real}, u, p, t)
    N = dimension(m)
    τ, thresh, reset, rest, Δt = p
    v, s = u[1:N], u[N+1:2N]
    vn = v + Δt * (-(v.-rest) ./ τ .+ m.W * s .+ randn(N))
    spikes = vn .> thresh
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