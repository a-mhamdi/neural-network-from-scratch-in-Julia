module MLP

using ActivationFunctions

export Layer, FeedForward, BackProp

mutable struct Layer
    W::Array{Float64, 2}
    b::Vector{Float64}
    act::Function
end

function Layer(input_size::Int, output_size::Int, act::Function)
    W = randn(output_size, input_size)
    b = zeros(output_size) # randn(output_size)
    Layer(W, b, act) 
end

function (l::Layer)(x::Array{Float64, 1})
    l.W * x .+ l.b
end

function FeedForward(layers::Vector{Layer}, x::Array{Float64, 1})
    # foldl((x, l) -> l.act.(l(x)), layers, init=x)
    a = [x]
    h = [x]
    for l in layers
        push!(a, l(a[end]))
        push!(h, l.act.(a[end]))
    end
    a, h
end

function BackProp(layers::Vector{Layer}, a::Array{Array{Float64, 1}, 1}, h::Array{Array{Float64, 1}, 1}, y::Array{Float64, 1})
    # Compute δ
    act_prime = eval(Symbol(layers[end].act, "_prime")).(a[end])
    ϵ = (y .- h[end])
    loss = 1/2 * sum(ϵ .^ 2)
    δ = [-(y .- h[end]) .* act_prime]
    for i in length(layers)-1:-1:1
        act_prime = eval(Symbol(layers[i].act, "_prime")).(a[i+1])
        pushfirst!(δ, (layers[i+1].W' * δ[1]) .* act_prime)
    end
    # Compute gradients
    ∇W = [δ[i] * h[i]' for i in 1:length(layers)]
    ∇b = δ
    loss, ∇W, ∇b
end

end # END



