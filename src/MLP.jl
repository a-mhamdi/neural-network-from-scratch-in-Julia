module MLP

using ActivationFunctions

export Layer, FeedForward, BackProp

export Regularizer

mutable struct Layer
    W::Array{Float64, 2}
    b::Vector{Float64}
    act::Function
end

function Layer(input_size::Int, output_size::Int, act::Function; distribution::Char='U')
    if act in [relu, leaky_relu] # He initialization
        if distribution == 'U' # uniform
            W = sqrt(6/input_size) .* (2 .* rand(output_size, input_size) .- 1)
        elseif distribution == 'N' # normal
            W = sqrt(2/input_size) .* randn(output_size, input_size)
        else
            @error "Invalid distribution"
        end
    else # Xavier (Glorot) initialization
        if distribution == 'U'
            W = sqrt(6/(input_size + output_size)) .* (2 .* rand(output_size, input_size) .- 1) 
        elseif distribution == 'N'
            W = sqrt(2/(input_size + output_size)) .* randn(output_size, input_size)
        else
            @error "Invalid distribution"
        end
    end
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

    Regularizer(a, b, c, d) = new(a, b, c, d)
    Regularizer() = new(:None, 0., 0., 0.)
end

function BackProp(layers::Vector{Layer}, A, H, data_x, data_y, reg) # ::Array{Array{Float64, 1}, 1}, h::Array{Array{Float64, 1}, 1}, y::Vector{Float64}, reg::Regularizer)
    
    loss, ∇W, ∇b = 0., [], []

    ∇W = [zeros(size(l.W)) for l in layers]
    ∇b = [zeros(size(l.b)) for l in layers]

    batch_size = size(A)[1]

    for i in 1:batch_size
        a = A[i]
        h = H[i]
        pushfirst!(h, data_x[i, :])
        y = data_y[i]

        # Compute δ
        act_prime = eval(Symbol(layers[end].act, "_prime")).(a[end])

        ϵ = (y .- h[end])
        loss = 1/2 * sum(ϵ .^ 2)

        if reg.method == :L1
            for l in layers
                loss += reg.λ .* sum(abs.(l.W))
            end
        elseif reg.method == :L2
            for l in layers
                loss += reg.λ/2 .* sum(l.W .^ 2)
            end
        elseif reg.method == :ElasticNet
            for l in layers
                loss += reg.r * reg.λ .* sum(abs.(l.W)) .+ (1-reg.r) * reg.λ/2 .* sum(l.W .^ 2)
            end
        end

        δ = [-(y .- h[end]) .* act_prime]
        for j in length(layers)-1:-1:1
            act_prime = eval(Symbol(layers[j].act, "_prime")).(a[j+1])
            pushfirst!(δ, (layers[j+1].W' * δ[1]) .* act_prime)
        end
        # Compute gradients

        ∇W .+= [δ[k] * h[k]' for k in 1:length(layers)]
        ∇b .+= δ

        a, h = [], []
    end

    ∇W ./= batch_size
    ∇W ./= batch_size

    loss, ∇W, ∇b
end

end # END



