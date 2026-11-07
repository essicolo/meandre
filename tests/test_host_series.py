"""Les series jours x troncons du domaine provincial restent en memoire vive : seule la tranche d'un bloc passe sur la carte."""
import numpy as np
import pytest
import torch

from meandre.run.province_data import HostSeries, check_host_series


def test_slice_goes_to_device_whole_series_stays_on_host():
    host = np.arange(24, dtype=np.float32).reshape(6, 4)
    s = HostSeries(host, "cpu")
    assert s.shape == (6, 4) and len(s) == 6
    block = s[2:4]
    assert torch.is_tensor(block) and block.shape == (2, 4)
    assert torch.equal(block, torch.from_numpy(host[2:4]))
    assert s._host is host


def test_whole_tensor_is_refused():
    check_host_series(forcing=HostSeries(np.zeros((3, 2), np.float32), "cpu"), et_obs=None)
    with pytest.raises(TypeError, match="et_obs"):
        check_host_series(et_obs=torch.zeros(3, 2))
