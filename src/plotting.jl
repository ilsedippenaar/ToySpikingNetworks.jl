using CairoMakie
using LinearAlgebra: eigen, normalize
using VoronoiCells: Rectangle, voronoicells

function eventplot!(ax::Axis, events::Vector{Vector{T}}; attrs...) where T <: Real
    y = range(0, 1, length(events)+1)
    for (events, ymin, ymax) in zip(events, y[1:end-1], y[2:end])
        vlines!(ax, events; ymin=ymin, ymax=ymax, attrs...)
    end
end

function summaryplot(network::Network)
	fig = Figure(resolution=(1000, 600))

	ax1, _ = scatter(fig[1, 1], network.positions)
	ax1.title = "Neuron Locations"
	ax1.aspect = DataAspect()
	ax1.xgridvisible = false
	ax1.ygridvisible = false

	conn_sub = fig[1, 2] = GridLayout()
	ax2, h = heatmap(
		conn_sub[1, 1],
		network.adj[end:-1:1, :]',  # make heatmap display the matrix as we see it
		colormap=Reverse(:diverging_gkr_60_10_c40_n256),
		colorrange=[-1, 1] * maximum(abs.(network.adj)),
	)
	ax2.title = "Connectivity Matrix"
	ax2.aspect = DataAspect()
	ax2.xlabel = "To"
	ax2.ylabel = "From"
	ax2.ylabelrotation = 0
	hidedecorations!(ax2, label = false)
	Colorbar(conn_sub[1, 2], h)

	fig
end

function dynamicsplot(A::StateSpaceSet, t; sample_traces = 5)
	N = size(A, 2) ÷ 2
	v = potentials(A)
	s = spikes(A)

	fig = Figure(resolution=(1200, 600))
	eventplot!(
		Axis(
			fig[1, 1],
			limits=(nothing, (0, N)),
			title="Spike train",
		),
		s,
		color=:black,
	)
	lines!(
		Axis(fig[1, 2], title="Spike count distribution"),
		sum.(s) |> sort |> reverse,
		linewidth=3,
	)
	
	ax = Axis(fig[2, 1:2], title="Sample traces")
	for i=1:sample_traces
		lines!(ax, t, v[:, i])
	end
	fig
end


function scaleboundingrect(p::Vector{<:Point2{T}}, s::Real) where T
	r = Rect(p)
	# the scale function (Base.:*) is kinda broken in GeometryBasics.jl?
	r = Rect(r.origin - r.widths .* (s-1)/2, r.widths .* s)
	r |> extrema .|> Point2{T} |> splat(Rectangle)
end

function voronoianimation(network::Network, A::StateSpaceSet, tvec)
	tess = voronoicells(network.positions, scaleboundingrect(network.positions, 1.1))
	v = potentials(A)
	i = Observable(tvec |> eachindex |> first)
	t = @lift tvec[$i]
	colors = get(cgrad([:black, :white]), Matrix(v), :extrema)
	fig = Figure()
	ax = Axis(fig[1, 1], title = @lift "t = $($t)")
	for (c, cell) in enumerate(tess.Cells)
		poly!(
			ax, cell, strokecolor=:black, strokewidth=0.5,
			color=@lift colors[$i, c]
		)
	end
	scatter!(network.positions, color=:black, markersize=3)
	CairoMakie.Makie.Record(fig, eachindex(tvec)) do i_
		i[] = i_
	end
end