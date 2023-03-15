using CairoMakie
using LinearAlgebra: eigen, normalize
using VoronoiCells: Rectangle, voronoicells

export dynamicsplot, summaryplot, voronoianimation, metricsplot

function eventplot!(ax::Axis, events::Vector{Vector{T}}; attrs...) where T <: Real
    y = range(0, 1, length(events)+1)
    for (events, ymin, ymax) in zip(events, y[1:end-1], y[2:end])
        vlines!(ax, events; ymin=ymin, ymax=ymax, attrs...)
    end
end

function summaryplot(network::Network)
    fig = Figure(resolution=(1000, 500))

    ax1, _ = scatter(fig[1, 1], network.pos)
    ax1.title = "Neuron Locations"
    ax1.aspect = DataAspect()
    ax1.xgridvisible = false
    ax1.ygridvisible = false

    conn_sub = fig[1, 2] = GridLayout()
    ax2, h = heatmap(
        conn_sub[1, 1],
        network.W[end:-1:1, :]',  # make heatmap display the matrix as we see it
        colormap=Reverse(:diverging_gkr_60_10_c40_n256),
        colorrange=[-1, 1] * maximum(abs.(network.W)),
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
            title="Spike trains",
        ),
        s,
        color=:black,
    )
    hist!(
        Axis(fig[1, 2], title="Spike count distribution"),
        sum.(s),
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

function _voronoianimation(network::Network, A::StateSpaceSet, tvec)
    tess = voronoicells(network.pos, scaleboundingrect(network.pos, 1.1))
    v = potentials(A)
    i = Observable(tvec |> eachindex |> first)
    t = @lift tvec[$i]
    colors = get(cgrad([:black, :white]), Matrix(v), :extrema)
    fig = Figure()
    ax = Axis(fig[1, 1], title = @lift "t = $($t)")
    hidedecorations!(ax)
    for (c, cell) in enumerate(tess.Cells)
        poly!(
            ax, cell, strokecolor=:black, strokewidth=0.5,
            color=@lift colors[$i, c]
        )
    end
    scatter!(network.pos, color=:black, markersize=3)
    update(i_) = i[] = i_
    fig, update
end

function voronoianimation(network::Network, A::StateSpaceSet, tvec; kwargs...)
    fig, update = _voronoianimation(network, A, tvec)
    CairoMakie.Makie.Record(update, fig, eachindex(tvec); kwargs...)
end

function voronoianimation(network::Network, A::StateSpaceSet, tvec, path; kwargs...)
    fig, update = _voronoianimation(network, A, tvec)
    record(update, fig, path, eachindex(tvec); kwargs...)
end

function metricsplot(metrics)
    xlabel_map = Dict(
        :N => L"N",
        :σᵦ => L"\sigma_\beta",
        :σₜₐᵤ => L"\sigma_\tau",
        :𝓲 => L"\mathcal{I}",
    )
    ylabel_map = Dict(
        :λ => L"\lambda",
        :r² => L"R^2",
    )
    fig = Figure(;resolution=(1600, 1200))
    yaxes = Dict()
    for (j, (v_name, v)) in enumerate(pairs(metrics))
        x = v.vals
        axes = []
        for (i, m) in enumerate((:r², :λ))
            y = v.metrics[m]
            ax = Axis(
                fig[i, j],
                ylabel=ylabel_map[m],
                ylabelrotation=0,
                xlabel=xlabel_map[v_name],
                ylabelsize=24,
                xlabelsize=24,
                xticks=(1:length(x), string.(x)),
            )
            boxplot!(
                ax,
                repeat(1:length(x), inner=size(y, 1)),
                reshape(y, :),
            )
            push!(axes, ax)
            push!(get!(yaxes, i, []), ax)
        end
        linkxaxes!(axes...)
    end
    for axs in values(yaxes)
        linkyaxes!(axs...)
    end
    fig
end
