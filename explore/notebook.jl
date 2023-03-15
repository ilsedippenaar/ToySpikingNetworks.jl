### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 43020c9a-c2d5-11ed-0b34-37b0e8651307
begin
    import Pkg; Pkg.add(url="https://github.com/ilsedippenaar/ToySpikingNetworks.jl")
end

# ╔═╡ 0a84568e-22f4-4520-9896-a056682e851b
using ToySpikingNetworks

# ╔═╡ e77e286a-1246-4329-aa4d-40b5083be867
ds, A, t = neuraldynamics(N=100, α=0.25, 𝓲=0.2, T=500);

# ╔═╡ 86ede83e-a673-4d48-b2f7-53c3fe9a372f
summaryplot(ds.f)

# ╔═╡ 4fdc5229-fd4a-43b8-9039-c69f571eeef0
dynamicsplot(A, t, sample_traces=3)

# ╔═╡ 24370eb1-3e1b-4cc6-8e13-466c486761a8
voronoianimation(ds.f, A, t)

# ╔═╡ 7e0388e9-285a-48ff-ba38-03990e3fef39
metrics = runanalysis(10, 1000);  # take smaller sample size for runtime purposes

# ╔═╡ da34aae5-5031-461a-b124-cd847275717a
metricsplot(metrics)

# ╔═╡ Cell order:
# ╠═43020c9a-c2d5-11ed-0b34-37b0e8651307
# ╠═0a84568e-22f4-4520-9896-a056682e851b
# ╠═e77e286a-1246-4329-aa4d-40b5083be867
# ╠═86ede83e-a673-4d48-b2f7-53c3fe9a372f
# ╠═4fdc5229-fd4a-43b8-9039-c69f571eeef0
# ╠═24370eb1-3e1b-4cc6-8e13-466c486761a8
# ╠═7e0388e9-285a-48ff-ba38-03990e3fef39
# ╠═da34aae5-5031-461a-b124-cd847275717a
