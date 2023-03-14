module SpikingNetworks

include("network.jl")
include("dynamics.jl")
include("plotting.jl")
include("experiments.jl")

export Network, NeuralModel
export waxman, dalesprinciple!
export eventplot!, dynamicsplot, summaryplot
export neuraldynamics

end
