module Metrics

export loss_fct
export r2_score
export cm, accuracy_score, precision_score, recall_score, f1_score


### LOSS FUNCTION
function loss_fct(y, ŷ; loss::Union{Symbol,String}=:mse)
    len = size(y, 1)
    if (isa(loss, Symbol) && loss == :mae) || (isa(loss, String) && lowercase(loss) == "mae") # mean absolute error
        ls = sum(abs.(y .- ŷ)) / len
    elseif (isa(loss, Symbol) && loss == :mse) || (isa(loss, String) && lowercase(loss) == "mse") # mean squared error
        ls = sum((y .- ŷ) .^ 2) / len
    elseif (isa(loss, Symbol) && loss == :rmse) || (isa(loss, String) && lowercase(loss) == "rmse") # root mean squared error
        ls = √(sum((y .- ŷ) .^ 2)) / len
    elseif (isa(loss, Symbol) && loss == :binarycrossentropy) || (isa(loss, String) && lowercase(loss) == "binarycrossentropy")
        ls = sum(-y .* log.(ŷ) .- (1 .- y) .* log.(1 .- ŷ)) / len # binary cross entropy
    elseif (isa(loss, Symbol) && loss == :crossentropy) || (isa(loss, String) && lowercase(loss) == "crossentropy")
        ls = sum(-y .* log.(ŷ)) / len # categorical cross entropy
    end

    ls
end

### === REGRESSION === ###

## COEFFICIENT OF DETERMINATION: R-SQUARED
function r2_score(y, ŷ)
    ȳ = sum(y, dims=1) ./ size(y, 1)
    SSres = sum((y .- ŷ) .^ 2, dims=1)
    SStot = sum((y .- ȳ) .^ 2, dims=1)
    r2 = 1 .- SSres ./ SStot
    
    printstyled("R-squared = $(r2)\n"; bold=true, color=:blue)
    r2
end

### === CLASSIFICATION === ###

function pn(y, ŷ) # 0: negative, 1: positive
    n_classes = size(y, 2)
    if n_classes == 1
        n_classes = 2
        y = hcat(y, 1 .- y)
        ŷ = hcat(ŷ, 1 .- ŷ)
    end

    tp = zeros(Int, n_classes)
    tn = zeros(Int, n_classes)
    fp = zeros(Int, n_classes)
    fn = zeros(Int, n_classes)

    for i in 1:n_classes
        tp[i] = sum(Int.(y[:, i] .== 1) .& (ŷ[:, i] .== 1))
        tn[i] = sum(Int.(y[:, i] .== 0) .& (ŷ[:, i] .== 0))
        fp[i] = sum(Int.(y[:, i] .== 0) .& (ŷ[:, i] .== 1))
        fn[i] = sum(Int.(y[:, i] .== 1) .& (ŷ[:, i] .== 0))
    end

    tp, tn, fp, fn
end

## CONFUSION MATRIX
function cm(y, ŷ)
    n_classes = size(y, 2)
    if n_classes == 1
        n_classes = 2
        y = hcat(y, 1 .- y)
        ŷ = hcat(ŷ, 1 .- ŷ)
    end
    row = "| " * ' ' ^ 3 * " | " * join(["($i)" for i in 1:n_classes], " | ") * " |"
    sep_row = '-' ^ length(row)

    printstyled("Confusion Matrix\n"; bold=true, color=:red)
    printstyled("Row -> Actual & Column -> Predicted \n"; underline=true, color=:light_magenta)
    println(sep_row)
    println(row)
    println(sep_row)

    for i in 1:n_classes
        row = string.([Int(sum(y[:, i] .* ŷ[:, j])) for j in 1:n_classes])
        sep_cell = [' ' ^ (4-length(cell)) for cell in row]
        println("| ($i) | ", join(row .* sep_cell, "| "), '|')
        println(sep_row)
    end
end

## ACCURACY
function accuracy_score(y, ŷ)
    n_classes = size(y, 2)
    if n_classes == 1
        n_classes = 2
        y = hcat(y, 1 .- y)
        ŷ = hcat(ŷ, 1 .- ŷ)
    end
    tp, tn, fp, fn = pn(y, ŷ)

    acc = []
    for i in 1:n_classes
        push!(acc, (tp[i] + tn[i]) / (tp[i] + tn[i] + fp[i] + fn[i]))
    end

    printstyled("Accuracy = $(acc)\n"; bold=true, color=:blue)
    acc, sum(acc) / n_classes # accuracy, mean accuracy
end

## PRECISION
function precision_score(y, ŷ)
    n_classes = size(y, 2)
    if n_classes == 1
        n_classes = 2
        y = hcat(y, 1 .- y)
        ŷ = hcat(ŷ, 1 .- ŷ)
    end
    tp, _, fp, _ = pn(y, ŷ)

    pre = []
    for i in 1:n_classes
        push!(pre, tp[i] / (tp[i] + fp[i]))
    end

    printstyled("Precision = $(pre)\n"; bold=true, color=:blue)
    pre, sum(pre) / n_classes # precision, mean precision
end

## RECALL
function recall_score(y, ŷ)
    n_classes = size(y, 2)
    if n_classes == 1
        n_classes = 2
        y = hcat(y, 1 .- y)
        ŷ = hcat(ŷ, 1 .- ŷ)
    end
    tp, _, _, fn = pn(y, ŷ)

    rec = []
    for i in 1:n_classes
        push!(rec, tp[i] / (tp[i] + fn[i]))
    end

    printstyled("Recall = $(rec)\n"; bold=true, color=:blue)
    rec, sum(rec) / n_classes # recall, mean recall
end

## F1 SCORE
function f1_score(y, ŷ)
    p, _ = precision_score(y, ŷ)
    r, _ = recall_score(y, ŷ)

    f1 = 2 * (p .* r) ./ (p .+ r)

    printstyled("F1-score = $(f1)\n"; bold=true, color=:blue)
    f1, sum(f1) / length(f1) # f1_score, mean f1_score
end

end # END
