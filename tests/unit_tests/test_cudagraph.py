# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import MagicMock, patch

import pytest
import torch

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.cudagraph import wrap_with_cuda_graph
from torchtitan.trainer import Trainer


def test_cuda_graph_config_rejects_pipeline_parallelism() -> None:
    with pytest.raises(ValueError, match="do not support pipeline parallelism"):
        Trainer.Config(
            parallelism=ParallelismConfig(pipeline_parallel_degree=2),
        )

    config = Trainer.Config(
        training=TrainingConfig(disable_cuda_graphs=True),
        parallelism=ParallelismConfig(pipeline_parallel_degree=2),
    )
    assert config.training.disable_cuda_graphs


def test_cuda_graph_config_rejects_fsdp_always_reshard() -> None:
    with pytest.raises(ValueError, match="reshard_after_forward=always"):
        Trainer.Config(
            parallelism=ParallelismConfig(fsdp_reshard_after_forward="always"),
        )

    config = Trainer.Config(
        training=TrainingConfig(disable_cuda_graphs=True),
        parallelism=ParallelismConfig(fsdp_reshard_after_forward="always"),
    )
    assert config.training.disable_cuda_graphs


def test_cuda_graph_wrapper_clones_reused_output() -> None:
    class PassthroughCUDAGraphWrapper:
        def __init__(self, fn, example_inputs):
            self.fn = fn

        def __call__(self, *args):
            return self.fn(*args)

    graph_loss = torch.tensor(0.0)
    fwd_bwd = MagicMock(return_value=graph_loss)

    accumulated_losses = []
    with (
        patch("torchtitan.distributed.cudagraph.utils.device_type", "cuda"),
        patch("torch.cuda.is_available", return_value=True),
        patch.object(torch.version, "hip", None),
        patch(
            "torchtitan.distributed.cudagraph.CUDAGraphWrapper",
            PassthroughCUDAGraphWrapper,
        ),
    ):
        runner = wrap_with_cuda_graph(fwd_bwd)
        for value in (1.0, 2.0, 3.0):
            graph_loss.fill_(value)
            loss = runner(
                torch.ones(1),
                torch.ones(1),
                torch.tensor(1),
                {"positions": torch.ones(1)},
            )
            accumulated_losses.append(loss.detach())

    torch.testing.assert_close(
        torch.sum(torch.stack(accumulated_losses)), torch.tensor(6.0)
    )
    assert fwd_bwd.call_count == 3
    _, _, global_valid_tokens, extra_kwargs = fwd_bwd.call_args.args
    torch.testing.assert_close(global_valid_tokens, torch.tensor(1))
    torch.testing.assert_close(extra_kwargs["positions"], torch.ones(1))


@pytest.mark.parametrize(
    ("device_type", "cuda_available", "hip_version"),
    [
        ("cpu", False, None),
        ("cuda", False, None),
        ("cuda", True, "6.3"),
        ("xpu", False, None),
    ],
)
def test_cuda_graph_wrapper_is_noop_without_nvidia_cuda(
    device_type: str,
    cuda_available: bool,
    hip_version: str | None,
) -> None:
    fwd_bwd = MagicMock()

    with (
        patch("torchtitan.distributed.cudagraph.utils.device_type", device_type),
        patch("torch.cuda.is_available", return_value=cuda_available),
        patch.object(torch.version, "hip", hip_version),
        patch("torchtitan.distributed.cudagraph.logger.warning") as warning,
    ):
        runner = wrap_with_cuda_graph(fwd_bwd)

    assert runner is fwd_bwd
    warning.assert_called_once()
