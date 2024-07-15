module MLP

using ActivationFunctions

export Layer, FeedForward, BackProp

export Regularizer

mutable struct Layer
    W::Array{Float64, 2}
    b::Vector{Float64}
    act::Function
end

function Layer(input_size::Int, output_size::Int, act::Function)
    W = 1/sqrt(input_size) .* (2 .* rand(output_size, input_size) .- 1) # Xavier initialization
    b = zeros(output_size)
    Layer(W, b, act) 
end

function (l::Layer)(x) # ::Array{Float64, 2})
    l.W * x .+ l.b
end

function FeedForward(layers::Vector{Layer}, x) # ::Union{Vector{Float64}, Matrix{Float64}, Array{Float64, 2}})
    nrows, _ = size(x)
    A, a = [], []
    H, h = [], []
    for i in 1:nrows
        data = x[i, :]
        for l in layers
            data = l(data)
            push!(a, data)
            data = l.act.(data)
            push!(h, data)
        end
        push!(A, a); a = []
        push!(H, h); h = []
    end
    A, H # dims = (batch_size, first_layer_output_size ... last_layer_output_size)
end

struct Regularizer
    method::Symbol # :L1, :L2, :ElasticNet
    λ::Float64
    r::Float64
    dropout::Float64

    Regularizer() = new(:None, 0.0, 0.0, 0.0)
end

function BackProp(layers::Vector{Layer}, A, H, x, y, reg) # ::Array{Array{Float64, 1}, 1}, h::Array{Array{Float64, 1}, 1}, y::Vector{Float64}, reg::Regularizer)
    
    a = A[end]
    h = H[end]
    pushfirst!(h, x[end, :])
    y = y[end]

    # Compute δ
    act_prime = eval(Symbol(layers[end].act, "_prime")).(a[end])

    ϵ = (y .- h[end])
    loss = 1/2 * sum(ϵ .^ 2)

    if reg.method == :L1
        loss += reg.λ .* sum(abs.(layers[end].W))
    elseif reg.method == :L2
        loss += reg.λ/2 .* sum(layers[end].W .^ 2)
    elseif reg.method == :ElasticNet
        loss += reg.r * reg.λ .* sum(abs.(layers[end].W)) .+ (1-reg.r) * reg.λ/2 .* sum(layers[end].W .^ 2)
    end

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



