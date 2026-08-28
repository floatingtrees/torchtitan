# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from torchtitan.config import ParallelismConfig
from torchtitan.distributed.pipeline_parallel import _split_module
from torchtitan.models.deepseek_v4 import (
    deepseek_v4_configs,
    model_registry,
    pipeline_deepseek_v4,
)
from torchtitan.models.deepseek_v4.model import DeepSeekV4Model


class _RecordingLayer(nn.Module):
    def __init__(self, delta: float, *, hash_router: bool):
        super().__init__()
        self.delta = delta
        self.moe_enabled = True
        self.moe = SimpleNamespace(router=SimpleNamespace(hash=hash_router))
        self.last_input_ids = None

    def forward(self, x, input_ids, attention_masks, positions):
        self.last_input_ids = input_ids.detach().clone()
        return x + self.delta


class _RecordingHcHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input_shape = None

    def forward(self, x, hc_fn, hc_scale, hc_base):
        self.last_input_shape = tuple(x.shape)
        return x.sum(dim=2)


class _Scale(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


def _build_forward_harness() -> DeepSeekV4Model:
    torch.manual_seed(0)
    model = DeepSeekV4Model.__new__(DeepSeekV4Model)
    nn.Module.__init__(model)
    model.hc_mult = 2
    model.n_main_layers = 4
    model.tok_embeddings = nn.Embedding(16, 4)
    model.layers = nn.ModuleDict(
        {
            str(layer_id): _RecordingLayer(
                float(layer_id + 1),
                hash_router=layer_id < 2,
            )
            for layer_id in range(4)
        }
    )
    model.hc_head_fn = nn.Parameter(torch.ones(1))
    model.hc_head_base = nn.Parameter(torch.ones(1))
    model.hc_head_scale = nn.Parameter(torch.ones(1))
    model.hc_head = _RecordingHcHead()
    model.norm = _Scale(0.5)
    model.lm_head = nn.Linear(4, 8, bias=False)
    model._skip_lm_head = False
    return model


def test_deepseek_v4_pipeline_chunks_match_full_forward():
    model = _build_forward_harness()
    full_model = copy.deepcopy(model)
    first_stage = _split_module(
        model,
        ["tok_embeddings", "layers.0", "layers.1"],
    )
    last_stage = _split_module(
        model,
        [
            "layers.2",
            "layers.3",
            "hc_head",
            "hc_head_fn",
            "hc_head_base",
            "hc_head_scale",
            "norm",
            "lm_head",
        ],
    )

    tokens_BL = torch.tensor([[1, 2, 3], [4, 5, 6]])
    positions_BL = torch.arange(3).expand(2, -1)
    expected_BLV = full_model(tokens_BL, positions=positions_BL)

    intermediate_BLCD = first_stage(tokens_BL, positions=positions_BL)
    actual_BLV = last_stage(intermediate_BLCD, positions=positions_BL)

    assert intermediate_BLCD.shape == (2, 3, 2, 4)
    assert actual_BLV.shape == (2, 3, 8)
    torch.testing.assert_close(actual_BLV, expected_BLV)
    assert list(first_stage.layers) == ["0", "1"]
    assert list(last_stage.layers) == ["2", "3"]
    torch.testing.assert_close(
        first_stage.layers["0"].last_input_ids,
        tokens_BL,
    )
    torch.testing.assert_close(
        last_stage.layers["2"].last_input_ids,
        positions_BL,
    )
    assert last_stage.hc_head.last_input_shape == (2, 3, 2, 4)

    for param_name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
        assert getattr(first_stage, param_name) is None
        last_stage_param = getattr(last_stage, param_name)
        torch.testing.assert_close(last_stage_param, getattr(full_model, param_name))

    first_stage_state = set(first_stage.state_dict())
    last_stage_state = set(last_stage.state_dict())
    assert first_stage_state.isdisjoint(last_stage_state)
    assert first_stage_state | last_stage_state == set(full_model.state_dict())


def test_pipeline_deepseek_v4_assigns_model_specific_modules():
    model = _build_forward_harness()
    model_config = deepseek_v4_configs["debugmodel"]()
    parallel_dims = SimpleNamespace(pp=2)
    parallelism = ParallelismConfig(pipeline_parallel_degree=2)

    with mock.patch(
        "torchtitan.models.deepseek_v4.parallelize.pipeline_llm",
        return_value="pipeline",
    ) as pipeline_llm:
        result = pipeline_deepseek_v4(
            model,
            parallel_dims=parallel_dims,
            parallelism=parallelism,
            model_config=model_config,
        )

    assert result == "pipeline"
    delegated_parallelism = pipeline_llm.call_args.kwargs["parallelism"]
    assert delegated_parallelism.module_fqns_per_model_part == [
        ["tok_embeddings", "layers.0", "layers.1"],
        [
            "layers.2",
            "layers.3",
            "norm",
            "lm_head",
            "hc_head",
            "hc_head_fn",
            "hc_head_base",
            "hc_head_scale",
        ],
    ]
    assert model_registry("debugmodel").pipelining_fn is pipeline_deepseek_v4


def test_pipeline_deepseek_v4_rejects_hash_layer_after_first_stage():
    model = _build_forward_harness()
    model_config = deepseek_v4_configs["debugmodel"]()
    parallelism = ParallelismConfig(
        pipeline_parallel_degree=2,
        module_fqns_per_model_part=[
            ["tok_embeddings", "layers.0"],
            ["layers.1", "layers.2", "layers.3", "norm", "lm_head"],
        ],
    )

    with mock.patch(
        "torchtitan.models.deepseek_v4.parallelize.pipeline_llm",
    ) as pipeline_llm:
        with pytest.raises(ValueError, match="hash-routed layers on stage 0"):
            pipeline_deepseek_v4(
                model,
                parallel_dims=SimpleNamespace(pp=2),
                parallelism=parallelism,
                model_config=model_config,
            )

    pipeline_llm.assert_not_called()


def test_pipeline_deepseek_v4_rejects_hc_head_parameter_before_last_stage():
    model = _build_forward_harness()
    model_config = deepseek_v4_configs["debugmodel"]()
    parallelism = ParallelismConfig(
        pipeline_parallel_degree=2,
        module_fqns_per_model_part=[
            [
                "tok_embeddings",
                "layers.0",
                "layers.1",
                "hc_head_scale",
            ],
            ["layers.2", "layers.3", "norm", "lm_head"],
        ],
    )

    with mock.patch(
        "torchtitan.models.deepseek_v4.parallelize.pipeline_llm",
    ) as pipeline_llm:
        with pytest.raises(
            ValueError, match="direct parameters only on the last stage"
        ):
            pipeline_deepseek_v4(
                model,
                parallel_dims=SimpleNamespace(pp=2),
                parallelism=parallelism,
                model_config=model_config,
            )

    pipeline_llm.assert_not_called()
