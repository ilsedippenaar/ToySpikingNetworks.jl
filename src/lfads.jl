using Flux
using Flux: @functor, flip
using MLUtils: chunk, DataLoader
using Distributions
import StatsBase: kldivergence

struct BiRNN{T}
    f::T
    r::T
end
@functor BiRNN
BiRNN(T, (in, out)::Pair; kwargs...) = BiRNN(T(in => out; kwargs...), T(in => out; kwargs...))
(m::BiRNN)(xs::AbstractVector) = vcat.([m.f(x) for x in xs], flip(m.r, xs))
(m::BiRNN)(xs::AbstractArray{T,3}) where T = m(eachslice(xs, dims=3) |> collect)

struct BiEncoder{T}
    f::T
    r::T
end
@functor BiEncoder
BiEncoder(T, (in, out)::Pair; kwargs...) = BiEncoder(T(in => out; kwargs...), T(in => out; kwargs...))
(m::BiEncoder)(xs::AbstractVector) = vcat(last(m.f(x) for x in xs), last(m.r(x) for x in Iterators.reverse(xs)))
(m::BiEncoder)(xs::AbstractArray{T,3}) where T  = m(eachslice(xs, dims=3) |> collect)

struct ToDistribution{D} end
function (::ToDistribution{D})(x) where D
    @assert iszero(size(x, 1) % fieldcount(D))
    D.(chunk(x, fieldcount(D), dims=1)...)
end
function (::ToDistribution{Normal})(x)
    @assert iseven(size(x, 1))
    μ, logσ² = chunk(x, 2, dims=1)
    Normal.(μ, exp.(logσ²/2))
end

@Base.kwdef struct LFADSArgs
    input_size::Integer
    g_size::Integer = 16
    factor_size::Integer = 4
    u_size::Integer = 2

    g_encoder_size::Integer = 16
    c_encoder_size::Integer = 16
    generator_size::Integer = 16
    controller_size::Integer = 16
    dropout::Float32 = 0.1f0
end

struct LFADSGeneratorCell
    Wᵍ
    controller
    RNNᵍᵉⁿ_cell
    Wᶠ
    Wʳ
end
@functor LFADSGeneratorCell
function LFADSGeneratorCell(a::LFADSArgs)
    # for clarity
    args = (
        Wᵍ = a.g_size == a.generator_size ? identity : Dense(a.g_size => a.generator_size),
        controller = Chain(
            GRU(2a.c_encoder_size + a.factor_size => a.controller_size),  # RNNᶜᵒⁿ
            Dropout(a.dropout),
            Dense(a.controller_size => 2a.u_size),  # Wᵘ
            ToDistribution{Normal}(),
        ),
        RNNᵍᵉⁿ_cell = Flux.GRUCell(a.u_size => a.generator_size),
        Wᶠ = Chain(Dropout(a.dropout), Dense(a.generator_size => a.factor_size)),
        Wʳ = Dense(a.factor_size => a.input_size),
    )
    LFADSGeneratorCell(args...)
end
function Flux.Recur(m::LFADSGeneratorCell, ĝ₀)
    ĝ₀ = m.Wᵍ(ĝ₀)
    RNNᵍᵉⁿ = Flux.Recur(m.RNNᵍᵉⁿ_cell, ĝ₀)
    f₀ = m.Wᶠ(ĝ₀)
    Flux.Recur(f₀) do fₜ₋₁, Eᶜᵒⁿₜ
        uₜ = m.controller([Eᶜᵒⁿₜ; fₜ₋₁])
        ûₜ = rand.(uₜ)  # TODO: do we need the reparameterization trick?
        gₜ = RNNᵍᵉⁿ(ûₜ)
        fₜ = m.Wᶠ(gₜ)
        rₜ = Poisson.(exp.(m.Wʳ(fₜ)))
        fₜ, (;uₜ, rₜ)
    end
end

struct LFADSModel
    g_encoder
    c_encoder
    generator_cell
end
@functor LFADSModel
LFADSModel(a::LFADSArgs) = LFADSModel(
    Chain(
        BiEncoder(GRU, a.input_size => a.g_encoder_size),
        Dropout(a.dropout),
        Dense(2a.g_encoder_size => 2a.g_size),
        ToDistribution{Normal}(),
    ),
    BiRNN(GRU, a.input_size => a.c_encoder_size),
    LFADSGeneratorCell(a),
)
function (m::LFADSModel)(xs)
    g₀ = m.g_encoder(xs)
    ĝ₀ = rand.(g₀)
    Eᶜᵒⁿ = m.c_encoder(xs)
    gen = Flux.Recur(m.generator_cell, ĝ₀)
    output = [gen(Eᶜᵒⁿₜ) for Eᶜᵒⁿₜ in Eᶜᵒⁿ]
    (;g₀, u = getfield.(output, :uₜ), r = getfield.(output, :rₜ))
end

struct NormalAR1{T<:Real} <: Distribution{Univariate, Continuous}
    μ::Vector{T}
    logσ²ₚ::Vector{T}
    logτ::Vector{T}
end
NormalAR1(μ::Real, logσ²::Real, logτ::Real) = NormalAR1(([x] for x in promote(μ, logσ², logτ))...)
@functor NormalAR1
function kldivergence(P_θx::AbstractVector{Normal{T}}, P_θ::NormalAR1{T}) where T
    # TODO: numerical stability?
    μ, logσ²ₚ, logτ = P_θ.μ[1], P_θ.logσ²ₚ[1], P_θ.logτ[1]
    τ = exp(logτ)
    σ²ₚ = exp(logσ²ₚ)
    α = exp(-1/τ)
    σ²ₑ = σ²ₚ*(1-α^2)
    kldivergence(P_θx[1], Normal(μ, √σ²ₑ)) + sum(
        kldivergence.(P_θx[2:end], @. convolve(μ + α*(P_θx[1:end-1] - μ), Normal(0, √σ²ₑ)))
    )
end

struct LearnableNormal{T<:Real} <: Distribution{Univariate, Continuous}
    μ::Vector{T}
    logσ²::Vector{T}
end
@functor LearnableNormal
LearnableNormal(μ::Real, logσ²::Real) = (d = Normal(μ, √exp(logσ²)); LearnableNormal{eltype(d)}([d.μ], [log(var(d))]))
Normal(p::LearnableNormal) = Normal(p.μ[1], p.logσ²[1] |> exp |> sqrt)
kldivergence(p::Normal, q::LearnableNormal) = kldivergence(p, Normal(q))

struct LFADS
    model
    Pᵍ
    Pᵘ
end
@functor LFADS
function loss(lfads::LFADS, input::AbstractVector)  # input = time x [features x batch]
    g₀, u, r = lfads.model(input)
    input, u, r = cat(input..., dims=3), cat(u..., dims=3), cat(r..., dims=3)
    𝓛ˣ = mean(sum(logpdf.(r, input), dims=(1, 3)))
    𝓛ᴰᴸ = mean(sum(kldivergence.(g₀, lfads.Pᵍ), dims=1)) + mean(sum(mapslices(u -> kldivergence(u, lfads.Pᵘ), u, dims=3), dims=1))
    -(𝓛ˣ - 𝓛ᴰᴸ)
end


rates = [4, 5, 2, 5, 8, 7, 6, 3, 2, 1]
X = cat([rand(Poisson(r), 1, 4, 100) .|> Float32 for r in rates]..., dims=1)
X = eachslice(X, dims=3) |> collect

args = LFADSArgs(input_size=length(rates))
model = LFADSModel(args)
lfads = LFADS(model, LearnableNormal(0f0, 0f0), NormalAR1(0f0, -1f0, -1f0))

train_set = [X, X, X]

# g₀, u, r = lfads.model(X)

# TODO: Adam params
# TODO: training params (num_epoch)
# TODO: loss weight parameters
opt_state = Flux.setup(Adam(), lfads)
my_log = []
for epoch in 1:100
  losses = Float32[]
  Flux.trainmode!(lfads)
  for (i, data) in enumerate(train_set)
    𝓛, ∇ = Flux.withgradient(m->loss(m, data), lfads)
    push!(losses, 𝓛)
    if !isfinite(𝓛)
      @warn "loss is $𝓛 on item $i" epoch
    else
        Flux.update!(opt_state, lfads, ∇[1])
    end
    Flux.reset!(lfads)
  end...
  Flux.testmode!(lfads)
  acc = mean(abs.(rates .- mean(
    getfield.(cat(lfads.model(X).r..., dims=3)[:, 4, :], :λ),
    dims=2)))
  push!(my_log, (; acc, losses))
end

my_log

using CairoMakie
lines(getfield.(my_log, :acc))
lines()
lines(vcat(getfield.(my_log, :losses)...))
