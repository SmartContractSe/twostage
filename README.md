# SmartInspect

SmartInspect constructs a 233-token lexical graph, learns a shared ProNE embedding space, pairs each target with all seeds from the same vulnerability category, and performs ExactType GPU matching by using the smaller complete-function AST in each pair as the convolution kernel. A pair is reported as a match only when both similarity scores exceed their respective thresholds. The implementation is independent of the original project.

Python 3.10+ is required, together with an NVIDIA GPU and a CUDA-enabled PyTorch installation. Run the following commands in this directory:

```
python3 -m pip install torch==2.11.0 numpy==2.2.6 scipy==1.15.3 scikit-learn==1.7.2 tree-sitter==0.25.2 tree-sitter-solidity==1.2.13
python3 smartinspect.py
```

The built-in example contains one seed function and one target function obtained by renaming variables in the seed. In our test, it produces `status: ok`, `graph_score: 1.0`, AST `score: 1.0`, and `detected: true`.

To test custom functions, run:

```
python3 smartinspect.py --seed seed.sol --target target.sol --op A
```

Each file should contain one complete function. The `--seed` and `--target` options may be specified multiple times. The strict threshold pairs for A, B, and C are `(0.3, 0.2)`, `(0.7, 0.2)`, and `(0.3, 0.8)`, respectively.

For multiple vulnerability categories, import and call `detect(seeds, targets)`, where `seeds` is a dictionary mapping each vulnerability category to a list of function source-code strings.

The included example is intended only to verify that the method runs correctly. It does not include the experimental seed library or datasets.
