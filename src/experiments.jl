using Distributions: LogNormal, Normal
using Random: default_rng
using DynamicalSystems: DeterministicIteratedMap, trajectory

rescale(p::Vector{<:AbstractPoint}) = (r = Rect(p); [(p .- r.origin) ./ r.widths for p=p])

function neuraldynamics(
	rng = default_rng();
	N = 100,  # number of neurons
	positions = rand(2, N),  # the locations of the neurons
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
	# positions = hcat(sort(positions |> eachcol |> vec, lt=(x, y)->sum(x) < sum(y))...)
	W = waxman(positions, α, rand(rng, β_prior, 1, N))
	W = dalesprinciple!(rng, W, 𝓲)
	network = Network(W, positions)

	model = NeuralModel(network.adj)
	ds = DeterministicIteratedMap(
	    model,
	    [ones(N) * rest; zeros(N)],
	    (
			τ = rand(rng, τ_prior, N),
			thresh = thresh,
			reset = reset,
			rest = rest,
			Δt = Δt,
		)
	)
	A, t = trajectory(ds, T, Ttr=Ttr)
	network, A, t
end

