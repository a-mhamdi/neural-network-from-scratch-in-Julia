using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)

push!(LOAD_PATH, pwd() * "/src")

using Revise

using Preprocessing
using ActivationFunctions, MLP
using Metrics

## Hyperparameters
mutable struct Settings
    epochs::Int
    batch_size::Int

    Settings(epochs, batch_size) = batch_size ≥ 1 ? new(epochs, batch_size) : error("Batch size must be greater than 1")
    Settings(epochs) = new(epochs, 1)
end

hp = Settings(16, 12) 

using RDatasets
iris = dataset("datasets", "iris")
x = iris[1:end, 1:end-1] |> Array; num_features = size(x)[2]
species = map( x -> if x=="setosa" x=1 elseif x=="versicolor" x=2 elseif x=="virginica" x=3 end, iris.Species); num_targets = maximum(levels(species))
y = zeros(length(species), num_targets)
y[species .== 1, 1] .= 1; y[species .== 2, 2] .= 1; y[species .== 3, 3] .= 1

(x_train, y_train), (x_test, y_test), (x_val, y_val) = data_split(x, y, train_size=.7, val_size=.1)
data_x = data_loader(x_train, hp.batch_size)
data_y = data_loader(y_train, hp.batch_size)

## Model architecture
model = [ # MLP
    Layer(num_features, 32, relu; distribution='n'),
    Layer(32, num_targets, softmax, distribution='n')
    ]

## Regularizer
regularizer = Regularizer(:none, .2, .6, .0) # method, λ, r, dropout

## Solver
solver = Solver(:crossentropy, :sgd, .03, regularizer)

ltrn, ltst = [], []

for epoch in 1:hp.epochs
    printstyled("=================== EPOCH #$epoch =====================\n"; bold=true, color=:red)
    for (data_in, data_out) in zip(data_x, data_y)
        TrainNN(model, data_in, data_out, x_val, y_val; solver)
    end

    ### TRAIN LOSS
    ŷ_train = Predict(model, x_train)
    loss = loss_fct(y_train, ŷ_train; loss=solver.loss)
    push!(ltrn, loss)

    ### TEST LOSS
    ŷ_test = Predict(model, x_test)
    loss = loss_fct(y_test, ŷ_test; loss=solver.loss)
    push!(ltst, loss)

    printstyled("*** @ last *** "; bold=true, color=:green)
    println("train loss: $(ltrn[end]) *** test loss: $(ltst[end])")
end

using Plots
plot(ltrn, label="train", xlabel="epoch", ylabel="loss", title="loss values")
p = plot!(ltst, label="test")
display(p)

ŷ_tst = Predict(model, x_test)
ŷ_tst = Int.(ŷ_tst .== maximum(ŷ_tst, dims=2))

## Confusion Matrix
cm(y_test, ŷ_tst)

## Accuracy Score
accuracy_score(y_test, ŷ_tst);

## F1 Score
f1_score(y_test, ŷ_tst);