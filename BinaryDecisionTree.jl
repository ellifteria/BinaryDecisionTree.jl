module BinaryDecisionTree

using DataFrames

export OUTPUT, OutputLevel, gini, gini_k, build_binary_decision_tree, print_btn

@enum OUTPUT begin
    no_output = 0
    some_output = 1
    full_output = 2
end

OutputLevel = Union{OUTPUT, Int64}

OptionalString = Union{String, Nothing}
OptionalFloat64 = Union{Float64, Nothing}
OptionalInt64 = Union{Int64, Nothing}
OptionalIntIntTuple = Union{Tuple{Int64, Int64}, Nothing}

mutable struct BinTreeNode
    split::OptionalString
    err::OptionalFloat64
    class_division::OptionalIntIntTuple
    class::OptionalInt64
    class_likelihood::OptionalFloat64
    node_0::Union{BinTreeNode, Nothing}
    node_1::Union{BinTreeNode, Nothing}
end


function gini(N::DataFrame)::Float64
    len = length(N[:, end])
    len_0 = length(N[N[:, end] .== 0, end])
    len_1 = length(N[N[:, end] .== 1, end])
    res = 1 - (len_0 / len)^2 - (len_1 / len)^2
    return res
end

function gini(N::DataFrame, N1::DataFrame, N2::DataFrame)::Float64
    len_N = length(N[:, end])
    len_N1 = length(N1[:, end])
    len_N2 = length(N2[:, end])
    res = len_N1/len_N * gini(N1) + len_N2/len_N * gini(N2)
    return res
end

function gini_k(N::DataFrame, k::Int64)::Float64
    global FULL_OUTPUT
    len = length(N[:, end])
    len_0 = length(N[N[:, end] .== 0, end])
    len_1 = length(N[N[:, end] .== 1, end])
    res = 1 - (len_0 / len)^k - (len_1 / len)^k
    return res
end

function gini_k(N::DataFrame, N1::DataFrame, N2::DataFrame, k::Int64)::Float64
    global FULL_OUTPUT
    len_N = length(N[:, end])
    len_N1 = length(N1[:, end])
    len_N2 = length(N2[:, end])
    res = len_N1/len_N * gini_k(N1, k) + len_N2/len_N * gini_k(N2, k)
    return res
end

function calc_gain(func::Function, N::DataFrame, N1::DataFrame, N2::DataFrame, output::OutputLevel = no_output)::Float64
    of_node = func(N)
    of_split = func(N, N1, N2)
    if output == full_output
        println("of node: $(round(of_node, digits=3)) \nof split: $(round(of_split, digits=3))")
    end
    gain = of_node - of_split
    return gain
end

function calc_gains(func::Function, N::DataFrame, ignore::Vector{Int64} = Vector{Int64}(), output::OutputLevel = no_output)::Int64
    num_indep = ncol(N) - 1
    max_gain = typemin(Float64)
    max_gain_i = -1
    for i = 1 : num_indep
        if !(i in ignore)
            if output == full_output
                println("\n\t$(names(N)[i]): calculating gain")
            end
            gain_for_i = calc_gain(func, N, N[N[:, i] .== 0, :], N[N[:, i] .== 1, :], output)
            if gain_for_i > max_gain
                max_gain = gain_for_i
                max_gain_i = i
            end
        end
    end
    if output == full_output
        println("\n\t$(names(N)[max_gain_i]) <- maximum gain")
    end
    return max_gain_i
end

function count_num_per_class(N::DataFrame)::Tuple{Int64, Int64}
    len_0 = length(N[N[:, end] .== 0, end])
    len_1 = length(N[N[:, end] .== 1, end])
    return len_0, len_1
end

function determine_class(class_division::Tuple{Int64, Int64})::Tuple{Int64, Float64}
    total = sum(class_division)
    pct_0 = class_division[1] / total
    pct_1 = class_division[2] / total
    if pct_0 > pct_1
        return 0, pct_0
    else
        return 1, pct_1
    end
    return class
end

function build_binary_decision_tree(func::Function, N::DataFrame, output::OutputLevel = full_output, ignore::Vector{Int64} = Vector{Int64}())::BinTreeNode
    curr_err = func(N)
    class_division = count_num_per_class(N)
    class, class_likelihood = determine_class(class_division)
    if (curr_err == 0) || (length(N[end, :]) - 1 == length(ignore))
        return BinTreeNode(nothing, curr_err, class_division, class, class_likelihood, nothing, nothing)
    end
    index_for_split = calc_gains(func, N, ignore, output)
    split_name = names(N)[index_for_split]
    if Int(output) > Int(no_output)
        println("\n!!! $(split_name): branching here !!!")
    end
    N0 = N[N[:, index_for_split] .== 0, :]
    N1 =  N[N[:, index_for_split] .== 1, :]
    func(N, N0, N1)
    if Int(output) > Int(no_output)
        println("\n↓↓↓ $(split_name)=0: beginning branch ↓↓↓")
    end
    node_0 = build_binary_decision_tree(func, N0, output, [ignore; index_for_split])
    if Int(output) > Int(no_output)
        println("↑↑↑ $(split_name)=0: branch complete ↑↑↑")
        println("\n↓↓↓ $(split_name)=1: beginning branch ↓↓↓")
    end
    node_1 = build_binary_decision_tree(func, N1, output, [ignore; index_for_split])
    if Int(output) > Int(no_output)
        println("↑↑↑ $(split_name)=1: branch complete ↑↑↑")
        println()
    end
    return BinTreeNode(split_name, curr_err, class_division, class, class_likelihood, node_0, node_1)
end

function print_btn(node::Union{BinTreeNode, Nothing}, level::Int64)::Nothing
    if isnothing(node)
        println()
        return
    end

    print(' '^(4 * level) * "|$(round(node.err, digits = 3)), $(node.class_division); ")
    println("CLASS=$(node.class): $(round(node.class_likelihood*100, digits=3))%")
    println(' '^(4 * level) * "|split by $(node.split)")
    if !isnothing(node.node_0)
        println(' '^(4 * level) * "|==0")
        print_btn(node.node_0, level + 1)
    end
    if !isnothing(node.node_1)
        println(' '^(4 * level) * "|==1")
        print_btn(node.node_1, level + 1)
    end
    return
end

function print_btn(node::BinTreeNode)::Nothing
    print_btn(node, 0)
end
    
end
