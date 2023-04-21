# BinaryDecisionTree.jl

A simple Julia script for creating binary decisions trees from a data frame.

---

## Example usage

In the Julia REPL:

```julia
julia> include("BinaryDecisionTree.jl")
julia> include("TestData.jl")
julia> build_binary_decision_tree(gini, df) 
```

## Custom `gini` commands

In the Julia REPL:

```julia
julia> gini_3(N) = gini_k(N, 3)
julia> gini_3(N, N1, N2) = gini_k(N, N1, N2, 3)
julia> include("BinaryDecisionTree.jl")
julia> include("TestData.jl")
julia> build_binary_decision_tree(gini_3, df) 
```
