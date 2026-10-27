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


def test_integer_batchnorm_buffers_and_input_preservation(tmp_path):
    a = {"running_mean": torch.tensor([1., 3.]), "num_batches_tracked": torch.tensor(2)}
    b = {"running_mean": torch.tensor([5., 7.]), "num_batches_tracked": torch.tensor(6)}
    originals = [{k: v.clone() for k, v in sd.items()} for sd in [a, b]]
    out = perform_fedavg([(a, 1), (b, 3)], tmp_path / "g.pt")
    assert torch.allclose(out["running_mean"], torch.tensor([4., 6.]))
    assert out["num_batches_tracked"].dtype == torch.int64
    assert out["num_batches_tracked"].item() == 5
    for sd, original in zip([a, b], originals):
        assert all(torch.equal(sd[k], original[k]) for k in sd)


def test_fractional_trust_weights(tmp_path):
    out = perform_fedavg([({"w": torch.tensor([0.])}, 0.25),
                         ({"w": torch.tensor([4.])}, 0.75)], tmp_path / "g.pt")
    assert torch.allclose(out["w"], torch.tensor([3.]))


@pytest.mark.parametrize("weights", [(0, 0), (-1, 2), (float("nan"), 1), (float("inf"), 1), (1e308, 1e308)])
def test_invalid_weights_rejected_without_overwriting_model(tmp_path, weights):
    path = tmp_path / "g.pt"
    path.write_bytes(b"previous model")
    with pytest.raises(ValueError, match="weights"):
        perform_fedavg([({"w": torch.tensor([1.])}, w) for w in weights], path)
    assert path.read_bytes() == b"previous model"


@pytest.mark.parametrize("other", [{"different_key": torch.tensor([1.])}, {"w": torch.ones(2)}])
def test_incompatible_models_rejected(tmp_path, other):
    with pytest.raises(ValueError, match="keys and tensor shapes"):
        perform_fedavg([({"w": torch.ones(1)}, 1), (other, 1)], tmp_path / "g.pt")
