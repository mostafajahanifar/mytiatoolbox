# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# The Original Code is Copyright (C) 2021, TIA Centre, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****
"""Unit test package for vanilla CNN within toolbox."""

import numpy as np
import pytest
import torch

from tiatoolbox.models.architecture.vanilla import CNNModel
from tiatoolbox.utils.misc import model_to

ON_GPU = False


def test_functional():
    """Test for creating backbone."""
    backbones = [
        "alexnet",
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "wide_resnet50_2",
        "wide_resnet101_2",
        "densenet121",
        "densenet161",
        "densenet169",
        "densenet201",
        "googlenet",
        "mobilenet_v2",
        "mobilenet_v3_large",
        "mobilenet_v3_small",
    ]
    assert CNNModel.postproc([1, 2]) == 1

    b = 4
    h = w = 512
    samples = torch.from_numpy(np.random.rand(b, h, w, 3))
    for backbone in backbones:
        try:
            model = CNNModel(backbone, num_classes=1)
            model_ = model_to(on_gpu=ON_GPU, model=model)
            model.infer_batch(model_, samples, on_gpu=ON_GPU)
        except ValueError:
            raise AssertionError(f"Model {backbone} failed.")

    # skipcq
    with pytest.raises(ValueError, match=r".*Backbone.*not supported.*"):
        CNNModel("shiny_model_to_crash", num_classes=2)
