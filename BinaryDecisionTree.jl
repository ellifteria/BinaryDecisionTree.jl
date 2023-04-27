using DataFrames

global FULL_OUTPUT::Bool = true;

OptionalString = Union{String, Nothing}
OptionalFloat64 = Union{Float64, Nothing}

mutable struct BinTreeNode
    split::OptionalString
    gini::OptionalFloat64
    class::OptionalString
    node_0::Union{BinTreeNode, Nothing}
    node_1::Union{BinTreeNode, Nothing}
end


function gini(N::DataFrame)::Float64
    global FULL_OUTPUT
    len = length(N[:, end])
    len_0 = length(N[N[:, end] .== 0, end])
    len_1 = length(N[N[:, end] .== 1, end])
    res = 1 - (len_0 / len)^2 - (len_1 / len)^2
    if FULL_OUTPUT
        println("\tgini=$res")
    end
    return res
end

function gini(N::DataFrame, N1::DataFrame, N2::DataFrame)::Float64
    global FULL_OUTPUT
    len_N = length(N[:, end])
    len_N1 = length(N1[:, end])
    len_N2 = length(N2[:, end])
    res = len_N1/len_N * gini(N1) + len_N2/len_N * gini(N2)
    if FULL_OUTPUT
        println("\tgini of split=$res")
    end
    return res
end

function gini_k(N::DataFrame, k::Int64)::Float64
    global FULL_OUTPUT
    len = length(N[:, end])
    len_0 = length(N[N[:, end] .== 0, end])
    len_1 = length(N[N[:, end] .== 1, end])
    res = 1 - (len_0 / len)^k - (len_1 / len)^k
    if FULL_OUTPUT
        println("\tgini $k=$res")
    end
    return res
end

function gini_k(N::DataFrame, N1::DataFrame, N2::DataFrame, k::Int64)::Float64
    global FULL_OUTPUT
    len_N = length(N[:, end])
    len_N1 = length(N1[:, end])
    len_N2 = length(N2[:, end])
    res = len_N1/len_N * gini_k(N1, k) + len_N2/len_N * gini_k(N2, k)
    if FULL_OUTPUT
        println("\tgini of split $k=$res")
    end
    return res
end

function calc_gain(func::Function, N::DataFrame, N1::DataFrame, N2::DataFrame)::Float64
    global FULL_OUTPUT
    gain = func(N) - func(N, N1, N2)
    if FULL_OUTPUT
        println("\tgain=$gain")
    end
    return gain
end

function calc_gains(func::Function, N::DataFrame, ignore::Vector{Int64} = Vector{Int64}())::Int64
    global FULL_OUTPUT
    num_indep = ncol(N) - 1
    max_gain = typemin(Float64)
    max_gain_i = -1
    for i = 1 : num_indep
        if !(i in ignore)
            if FULL_OUTPUT
                println("\n\t$(names(N)[i]): calculating gain")
            end
            gain_for_i = calc_gain(func, N, N[N[:, i] .== 0, :], N[N[:, i] .== 1, :])
            if gain_for_i > max_gain
                max_gain = gain_for_i
                max_gain_i = i
            end
        end
    end
    if FULL_OUTPUT
        println("\n\t$(names(N)[max_gain_i]) <- maximum gain")
    end
    return max_gain_i
end

function build_binary_decision_tree(func::Function, N::DataFrame, ignore::Vector{Int64} = Vector{Int64}())::BinTreeNode
    curr_err = func(N)
    if (curr_err == 0) || (length(N[end, :]) - 1 == length(ignore))
        return BinTreeNode(nothing, curr_err, nothing, nothing, nothing)
    end
    index_for_split = calc_gains(func, N, ignore)
    split_name = names(N)[index_for_split]
    println("\n!!! $(split_name): branching here !!!")
    N0 = N[N[:, index_for_split] .== 0, :]
    N1 =  N[N[:, index_for_split] .== 1, :]
    func(N, N0, N1)
    println("\n↓↓↓ $(split_name)=0: beginning branch ↓↓↓")
    node_0 = build_binary_decision_tree(func, N0, [ignore; index_for_split])
    println("↑↑↑ $(split_name)=0: branch complete ↑↑↑")
    println("\n↓↓↓ $(split_name)=1: beginning branch ↓↓↓")
    node_1 = build_binary_decision_tree(func, N1, [ignore; index_for_split])
    println("↑↑↑ $(split_name)=1: branch complete ↑↑↑")
    println()
    return BinTreeNode(split_name, curr_err, nothing, node_0, node_1)
end
