import pytest
import torch
from h_node.ml import perform_fedavg


def test_fedavg_of_identical_models_is_unchanged(tmp_path):
    sd = {"w": torch.tensor([1.0, 2.0])}
    out = perform_fedavg([(sd, 10), (sd, 30)], tmp_path / "g.pt")
    assert torch.allclose(out["w"], sd["w"])


def test_fedavg_weights_by_sample_count(tmp_path):
    a = {"w": torch.tensor([0.0])}
    b = {"w": torch.tensor([4.0])}
    out = perform_fedavg([(a, 1), (b, 3)], tmp_path / "g.pt")
    assert torch.allclose(out["w"], torch.tensor([3.0]))


def test_fedavg_saves_global_weights(tmp_path):
    path = tmp_path / "g.pt"
    perform_fedavg([({"w": torch.tensor([1.0])}, 5)], path)
    assert torch.allclose(torch.load(path)["w"], torch.tensor([1.0]))


def test_fedavg_rejects_empty():
    with pytest.raises(ValueError):
        perform_fedavg([], "unused.pt")
