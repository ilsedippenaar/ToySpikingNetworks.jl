using Distributions: MvNormal, Exponential, LogitNormal, Gamma, MixtureModel, sample
using LinearAlgebra: I, norm, Symmetric, diagind
using Distances: Euclidean, pairwise
using Random: AbstractRNG, default_rng
using GeometryBasics: AbstractPoint

struct Network
    adj::Matrix
    positions::Vector{<:AbstractPoint}

    function Network(adj, positions)
        @assert size(adj, 1) == size(adj, 2)
        @assert size(adj, 1) == length(positions)
        new(adj, positions)
    end
end

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
