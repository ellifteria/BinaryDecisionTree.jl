# BinaryDecisionTree.jl

A simple Julia script for creating binary decisions trees from a data frame.

---

## Example usage

In the Julia REPL:

```julia
julia> include("BinaryDecisionTree.jl")
julia> include("TestData.jl")
julia> tree = build_binary_decision_tree(gini, df)
julia> print_btn(tree)
```

Output:

```
|0.489795918367347, (3, 4); CLASS=1: 0.5714285714285714
|split by F1
|==0
    |0.0, (1, 0); CLASS=0: 1.0
    |split by nothing
|==1
    |0.4444444444444444, (2, 4); CLASS=1: 0.6666666666666666
    |split by F3
    |==0
        |0.4444444444444445, (2, 1); CLASS=0: 0.6666666666666666
        |split by F2
        |==0
            |0.0, (1, 0); CLASS=0: 1.0
            |split by nothing
        |==1
            |0.5, (1, 1); CLASS=1: 0.5
            |split by nothing
    |==1
        |0.0, (0, 3); CLASS=1: 1.0
        |split by nothing
```

## Custom `gini` commands

In the Julia REPL:

```julia
julia> include("BinaryDecisionTree.jl")
julia> include("TestData.jl")
julia> gini_3(N) = gini_k(N, 3)
julia> gini_3(N, N1, N2) = gini_k(N, N1, N2, 3)
julia> tree = build_binary_decision_tree(gini_3, df)
julia> print_btn(tree)
```
