push!(LOAD_PATH, pwd() * "/src")

using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)

# Hyperparameters
struct Settings
    epochs::Int
    η::Float64
    batch_size::Int
end
hp = Settings(32, .01, 12) 

using Random
# Artificial data
n, α, β = 512, 1., -.5
x = 2 .* randn(n) .+ 5
y = α .* x .+ β

using Preprocessing
(x_train, y_train), (x_test, y_test), (x_val, y_val) = data_split(x, y, train_size=.7, val_size=.1)
data_ld = data_loader(x_train, y_train, hp.batch_size)

# Model architecture
using MLP, ActivationFunctions
model = [Layer(1, 1, relu)] # SLP

# Regularizer
regularizer = Regularizer(:None, 0.01, 0.5, 0.1)

coef = 1.
if regularizer.method == :L1
    # FIXME - L1 regularization 
elseif regularizer.method == :L2
    coef -= hp.η * optimizer.λ / hp.batch_size
elseif regularizer.method == :ElasticNet
    # FIXME - ElasticNet regularization
end

lvec = []
for epoch in 1:hp.epochs
    for data in data_ld
        # Forward pass
        a, h = FeedForward(model, data[1,:])
        # Backward pass
        loss, ∇W, ∇b = BackProp(model, a, h, data[2,:], regularizer)
        push!(lvec, loss)
        # Update parameters
        for i in 1:lastindex(model)
            model[i].W  = coef * model[i].W - hp.η * ∇W[i]
            model[i].b -= hp.η * ∇b[i]
        end
    end
    loss = sum( ( y_test .- (model[1].W .* x_test .+ model[1].b') ) .^ 2 ) / length(y_test)
    @info "Epoch: $epoch >>> Loss: $loss *** ̂α = $(model[1].W[1]) *** ̂β = $(model[1].b[1])"
end

using Plots
plot(lvec, legend=false, title="Loss Function J")
xlabel!("Iterations")
ylabel!("Loss")
