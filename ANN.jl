#ANN.jl

using BenchmarkTools

mutable struct NeuralNetwork
    numOfLayers::Int
    dimOfLayers::Vector{Int}
    Ws::Vector{Array{Float64, 2}}
    activateF::Function
    activateF_Diff::Function
end

function predict(nn::NeuralNetwork, x::Vector{Float64})
    layers = Vector{Array{Float64, 2}}(undef, nn.numOfLayers)
    layers[1] = x'
    for i = 2:nn.numOfLayers
        layers[i] = nn.activateF.(layers[i - 1] * nn.Ws[i - 1])
    end
    return layers[nn.numOfLayers]
end

function back_prop_learning(nn::NeuralNetwork, examples, α)
    a = Vector{Array{Float64, 2}}(undef, nn.numOfLayers)
    b = Vector{Array{Float64, 2}}(undef, nn.numOfLayers)
    Δ = Vector{Array{Float64, 2}}(undef, nn.numOfLayers)
    # a -- in
    # b -- out
    # delta -- error
    for (x, y) in examples
        b[1] = x'
        for i = 2:nn.numOfLayers
            a[i] = b[i - 1] * nn.Ws[i - 1]
            b[i] = nn.activateF.(a[i])
        end

        Δ[end] = @. nn.activateF_Diff(a[end]) * (y - b[end])
        for i = (nn.numOfLayers - 1):-1:1
            Δ[i] = nn.activateF_Diff.(a[i]) .* (nn.Ws[i] * Δ[i + 1]')'
        end

        for l = 1:(nn.numOfLayers - 1)
            nn.Ws[l] += α * b[l]' * Δ[l + 1]
        end
    end
end
