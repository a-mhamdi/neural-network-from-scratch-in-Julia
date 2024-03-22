#include("src/ActivationFunctions.jl")
#include("src/MLP.jl")

push!(LOAD_PATH, pwd() * "/src")

using ActivationFunctions
using MLP

# Hyperparameters
struct Settings
    epochs::Int
    η::Float64
    # batch_size::Int
end

using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)

hp = Settings(16, .01) 

using Random
# Artificial data
n, α, β = 512, 1., -.5
x = 2 .* randn(n) .+ 5
y = α .* x .+ β

# Model architecture
model = [Layer(1, 1, relu)] # SLP

lvec = []
for epoch in 1:hp.epochs
    for data in (zip(x, y)) 
        # Forward pass
        a, h = FeedForward(model, [data[1]])
        # Backward pass
        loss, ∇W, ∇b = BackProp(model, a, h, [data[2]])
        push!(lvec, loss)
        # Update parameters
        for i in 1:lastindex(model)
            model[i].W -= hp.η * ∇W[i]
            model[i].b -= hp.η * ∇b[i]
        end
        @info "Epoch: $epoch >>> Loss: $loss *** ̂α = $(model[1].W[1]) *** ̂β = $(model[1].b[1])"
    end
end

using Plots
plot(1:n*hp.epochs, lvec, legend=false, title="Loss Function J")