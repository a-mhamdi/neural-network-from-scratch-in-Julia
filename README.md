# Neural Network From Scratch In `Julia`
Without using any deep learning frameworks, we construct and train a neural network architecture in `Julia` from the ground up.

## Architecture's Design
The neural network (NN) is built from scratch and trained on some data. Here is a possible representation of the NN architecture:

```julia
model = [ # MLP
    Layer(num_features, num_neurons_1, relu; distribution='n'),
    Layer(num_neurons_1, num_neurons_2, relu; distribution='n'), 
    Layer(num_neurons_2, num_targets, softmax; distribution='n')
    ]
```
where `num_features`, `num_targets` and `num_neurons_\d+` denote, respectively, the numbers of input features, output targets, and neurons in the hidden layers. The choice of distribution for the weights initialization is either a normal distribution (`n`) or uniform distribution (`u`). Both `Xavier` and `He` methods were implemented.

Some of the hyperparameters are configured as follows:
```julia
Settings(epochs, batch_size)
```

## Model's Training
We can define a `struct` for the regularizer as follows:
```julia
Regularizer(method, λ, r, dropout)
```
`method` can be symbol or string of one of the following: `l1`, `l2`, `elasticnet`, or `none`. The `λ` parameter is the regularization parameter. The `r` parameter determines the mix of penalties in case of `elasticnet` method. The `dropout` parameter is the dropout rate. `loss` and `optimizer` are accessed through:
```julia
Solver(loss, optimizer, learning_rate, regularizer)
```
`loss` can be `:mae`, `:mse`, `:rmse`, `:binarycrossentropy` or `:crossentropy`. `:sgd` is the default optimizer. The model is trained using the following method:
```julia
TrainNN(model, data_in, data_out, x_val, y_val; solver)
```
Under the hood, the `TrainNN` method calls the `FeedForward` and `BackProp` functions. The `FeedForward` method returns the pre-activations and the hypothesis of the model: `data_cache`. The method signature is:
```julia
data_cache = FeedForward(model, data_in; solver::Solver)
```
The `BackProp` method allows to return the `loss` and the gradients of the weights and biases: `∇W` and `∇b`, as follows:
```julia
loss, ∇W, ∇b = BackProp(model, data_cache, data_out; solver::Solver)
```

Detailed steps of the backpropagation algorithm are shown in the images below ([source](https://github.com/a-mhamdi/jlai/blob/main/Slides-Labs/Demystifying%20AI%20Sorcery%20(Part-1).pdf)).

![BackProp](./Images/backprop.gif)

The code is written in `Julia`. The `main.jl` file contains the primary code to setup the simulation. The required modules are located in the `src` folder. 

## Simulation Outcomes

The figure below displays the model's *loss* for both the *training* and *test sets* at the end of each epoch.

![Loss](./Images/plot_loss.svg)

The following provides specifics about *confusion matrix*, *accuracy*, *precision*, *recall* and *f1-score* metrics.

![Metrics](./Images/results.png)

**Note:** The code is not optimized for performance. It is written for educational purposes. There is always room for improvement.

**TODO:** Implement the following features:
- [ ] parallelization of backprop on the batch of data instead of using a for loop;
- [ ] optimizers: `SGD+Momentum`, `SGD+Nesterov`, `Adam`, `RMSprop`, `Adagrad`, `Adadelta`.
