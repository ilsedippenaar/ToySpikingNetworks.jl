using Distributions: LogNormal, Normal
using Random: default_rng, seed!
using DynamicalSystems: DeterministicIteratedMap, trajectory
using ChaosTools: lyapunov
using StatsBase: mean

export neuraldynamics, runanalysis

rescale(p::Vector{<:AbstractPoint}) = (r = Rect(p); [(p .- r.origin) ./ r.widths for p=p])

function neuraldynamics(;
    rng = default_rng(),
    N = 100,  # number of neurons
    positions = rand(rng, 2, N),  # the locations of the neurons
    α = 0.15,  # α in the Waxman graph: sets spatial extent of connectivity
    β_prior = LogNormal(0, 1),  # distribution of β in the Waxman model
    𝓲 = 0.1,   # percent of neurons that have inhibitory output
    τ_prior = Normal(0.9, 0.0001),  # distribution of the time constant
    thresh = 0.6,  # threshold at which a spike occurs
    reset = -3,  # membrane potential after a spike occurs
    rest = 0,  # resting state of the network
    Δt = 0.1,  # time step in the dynamics evolution
    T = 1000,  # number of time steps to evolve the system
    Ttr = 10,  # amount of time to evolve the system before recording
)
    positions = positions |> eachcol .|> Point2 |> rescale |> sort
    W = waxman(positions, α, rand(rng, β_prior, 1, N))
    W = dalesprinciple!(rng, W, 𝓲)
    network = Network(W, positions)
    τ = rand(rng, τ_prior, N)
    ds = DeterministicIteratedMap(
        network,
        [ones(N) * rest; zeros(N)],
        (τ, thresh, reset, rest, Δt)
    )
    A, t = trajectory(ds, T, Ttr=Ttr)
    ds, A, t
end

function calcmetrics(;kwargs...)
    ds, A, _ = neuraldynamics(;kwargs...)
    v = potentials(A)
    X = Matrix(v[1:end-1])
    Y = Matrix(v[2:end])
    # julia doesn't have multivariate, multiresponse
    # linear regression available?????
    W = X \ Y
    r² = 1 - sum((X*W .- Y).^2) / sum((Y .- mean(Y)).^2)
    λ = lyapunov(ds, 1000, Ttr=10)
    (;r², λ)
end

function runanalysis(reps=100, T=5000)
    var_vals = (
        N = [25, 50, 100, 200],
        σᵦ = [0.01, 0.1, 1],
        σₜₐᵤ = [0.0001, 0.001, 0.01, 0.1],
        𝓲 = [0, 0.1, 0.2, 0.3, 0.4],
    )
    metric_names = (:r², :λ)
    seed!(1234)
    metrics = Dict()
    for (v, vals) in pairs(var_vals)
        v_metrics = []
        for val in vals
            kwargs = Dict{Symbol,Any}(:T => T)
            if v === :σᵦ
                kwargs[:β_prior] = LogNormal(0, val)
            elseif v === :σₜₐᵤ
                kwargs[:τ_prior] = Normal(0.9, val)
            elseif v === :𝓲
                kwargs[:𝓲] = val
            elseif v === :N
                kwargs[:N] = val
                # keep number neurons in neighborhood the same
                kwargs[:α] = 0.15 / √(val/100)
            else
                error("Unrecognized value: $v")
            end
            out = [calcmetrics(;kwargs...) for _ in 1:reps]
            push!(v_metrics, out)
        end
        v_metrics = hcat(v_metrics...)
        metrics[v] = (
            metrics = Dict(
                n => getfield.(v_metrics, n) for n in metric_names
            ),
            vals = vals,
        )
    end
    metrics
end

