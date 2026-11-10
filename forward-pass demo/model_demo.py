"""Run the FD001 model once on synthetic sensor windows."""

import torch

from transfer_hypergraph_model import HyperGraphNet


# FD001 model settings. Run this file directly; no arguments or data needed.
sensor_num = 14
seq_len = 30
d_model = 64
h_model = 32  # Structural embedding width in the FD001 implementation.
hyperedge_num = 5
k = 3
head = 4
dropout = 0.2
device = "auto"  # "auto", "cpu", or "cuda"


def main():
    selected_device = device
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    selected_device = torch.device(selected_device)
    torch.manual_seed(2026)

    model = HyperGraphNet(
        sensor_num=sensor_num,
        seq_len=seq_len,
        d_model=d_model,
        h_model=h_model,
        hyperedge_num=hyperedge_num,
        k=k,
        head=head,
        dropout=dropout,
        device=selected_device,
    ).eval()

    sensor_windows = torch.randn(2, seq_len, sensor_num, device=selected_device)
    with torch.inference_mode():
        predictions, structures = model(sensor_windows, return_structure=True)

    assert predictions.shape == (2, 1) and torch.isfinite(predictions).all()
    assert len(structures) == len(sensor_windows)
    for edge_index, edge_weight in structures:
        assert edge_index.shape == (2, sensor_num * k)
        assert edge_weight.shape == (sensor_num * k,)
        assert torch.isfinite(edge_weight).all()

    print(f"Device: {selected_device}")
    print(f"Synthetic input shape: {tuple(sensor_windows.shape)}")
    print(f"Untrained output shape: {tuple(predictions.shape)}")
    print(f"Retained node-hyperedge connections per sample: {structures[0][0].shape[1]}")


if __name__ == "__main__":
    main()
