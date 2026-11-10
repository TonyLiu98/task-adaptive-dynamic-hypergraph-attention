# FD001 hypergraph attention model

This package contains the PyTorch implementation of the proposed task-adaptive
dynamic hypergraph attention model for the C-MAPSS FD001 RUL task. It includes
the model source and a minimal forward-pass example using synthetic inputs.

- `transfer_hypergraph_model.py`: task-level hypergraph construction, two-step
  attention propagation, feature fusion, and RUL prediction layers.
- `model_demo.py`: constructs the model and checks the input/output shapes and
  number of retained connections. Its model settings can be edited at the top
  of the file.

## Run the example

```sh
python -m pip install -r requirements.txt
python model_demo.py
```

The example uses CUDA if available and otherwise runs on CPU. To force CPU,
set `device = "cpu"` in `model_demo.py`.

The FD001 defaults are a 30-cycle window, 14 sensor channels, five hyperedges,
top-`k` of 3, a model width of 64, four attention heads, and a dropout rate
of 0.2. The input shape is `[batch, time, sensors]`; the output
has one scalar per input window.

The example does not train the model or reproduce the reported performance
metrics. It does not use experimental data, and the model-code folder does
not include trained weights or training logs.

## Processed real-engine test data

[`processed_engine_testdata.csv`](../processed_engine_testdata.csv) contains one
processed flight record from an engine experiment. The nine model
inputs, in order, are `D8`, `a1`, `a2`, `PLA`, `H`, `M`, `V`, `n2`, and `n1`.
The prediction targets are `T1` and `T6`. The manuscript refers to `M` as `Ma`.

This file is separate from the C-MAPSS FD001 example. The real-engine
experiment uses 30 consecutive records per window with a stride of one.
The target values are taken from the last record of each window. The
separate training flight is not included.
