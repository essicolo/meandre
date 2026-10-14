"""Largeur du champ spatial declaree : field_hidden fixe les couches cachees et survit a la sauvegarde."""
import torch

from meandre.model import HydroModel


def test_field_hidden_sets_width_and_default_unchanged():
    m0 = HydroModel(n_nodes=5, n_territorial=4)
    m1 = HydroModel(n_nodes=5, n_territorial=4, field_hidden=64)
    assert m0.spatial_encoder.fc1.out_features == 256
    assert m1.spatial_encoder.fc1.out_features == 64
    n0 = sum(p.numel() for p in m0.spatial_encoder.parameters())
    n1 = sum(p.numel() for p in m1.spatial_encoder.parameters())
    assert n1 < n0


def test_field_hidden_saved_and_reloaded(tmp_path):
    m = HydroModel(n_nodes=5, n_territorial=4, field_hidden=64)
    f = tmp_path / "m.pt"
    m.save(f)
    r = HydroModel.from_checkpoint(f)
    assert r.spatial_encoder.fc1.out_features == 64
