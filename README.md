# BinaryDecisionTree.jl

A simple Julia script for creating binary decisions trees from a data frame.

---

## Example usage

In the Julia REPL:

```julia
julia> include("BinaryDecisionTree.jl"); include("TestData.jl");
julia> using .BinaryDecisionTree; using .BDTTestData;
julia> tree = build_binary_decision_tree(gini, test_df, 0);
julia> print_btn(tree)
```

Output:

```
|0.49, (3, 4); CLASS=1: 57.143%
|split by F1
|==0
    |0.0, (1, 0); CLASS=0: 100.0%
    |split by nothing
|==1
    |0.444, (2, 4); CLASS=1: 66.667%
    |split by F3
    |==0
        |0.444, (2, 1); CLASS=0: 66.667%
        |split by F2
        |==0
            |0.0, (1, 0); CLASS=0: 100.0%
            |split by nothing
        |==1
            |0.5, (1, 1); CLASS=1: 50.0%
            |split by nothing
    |==1
        |0.0, (0, 3); CLASS=1: 100.0%
        |split by nothing
```

## Custom `gini` commands

In the Julia REPL:

```julia
julia> include("BinaryDecisionTree.jl"); include("TestData.jl");
julia> using .BinaryDecisionTree; using .BDTTestData;
julia> gini_3(N) = gini_k(N, 3);
julia> gini_3(N, N1, N2) = gini_k(N, N1, N2, 3);
julia> tree = build_binary_decision_tree(gini_3, test_df, 0);
julia> print_btn(tree)
```

Output:

```
|0.735, (3, 4); CLASS=1: 57.143%
|split by F1
|==0
    |0.0, (1, 0); CLASS=0: 100.0%
    |split by nothing
|==1
    |0.667, (2, 4); CLASS=1: 66.667%
    |split by F3
    |==0
        |0.667, (2, 1); CLASS=0: 66.667%
        |split by F2
        |==0
            |0.0, (1, 0); CLASS=0: 100.0%
            |split by nothing
        |==1
            |0.75, (1, 1); CLASS=1: 50.0%
            |split by nothing
    |==1
        |0.0, (0, 3); CLASS=1: 100.0%
        |split by nothing
```
