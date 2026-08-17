# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch

import torchtitan.models.deepseek_v4.attention_gym as attention_gym_module
from torchtitan.models.deepseek_v4.attention_gym import (
    _selected_attention_sharding_config,
    attention_gym,
    AttentionGymSelectedAttention,
    SelectedAttentionKernel,
)
from torchtitan.models.deepseek_v4.config_registry import deepseek_large_debug_config


class TestDeepSeekV4SelectedAttentionOverride(unittest.TestCase):
    def test_large_debug_config_exercises_all_attention_types(self):
        model_config = deepseek_large_debug_config().model_spec.model

        self.assertEqual(model_config.compress_ratios, (1, 4, 128, 4))
        self.assertEqual(
            [layer.attention.n_heads for layer in model_config.layers],
            [128, 128, 128, 128],
        )

    def test_factory_adds_kernel_config(self):
        attention_config = (
            deepseek_large_debug_config().model_spec.model.layers[2].attention
        )

        with patch.object(
            attention_gym_module,
            "_ATTENTION_GYM_IMPORT_ERROR",
            None,
        ):
            replacement = attention_gym(attention_config)

        self.assertIsInstance(replacement, AttentionGymSelectedAttention.Config)
        self.assertEqual(
            replacement.selected_attention_kernel.sliding_window_size,
            attention_config.window_size,
        )
        self.assertEqual(replacement.selected_attention_kernel.backend, "cute")

    def test_factory_requires_attention_gym(self):
        attention_config = (
            deepseek_large_debug_config().model_spec.model.layers[0].attention
        )
        import_error = ImportError("No module named 'attn_gym'")

        with (
            patch.object(
                attention_gym_module,
                "_ATTENTION_GYM_IMPORT_ERROR",
                import_error,
            ),
            self.assertRaisesRegex(ImportError, "meta-pytorch/attention-gym"),
        ):
            attention_gym(attention_config)

    def test_kernel_delegates_to_selected_attention(self):
        kernel = SelectedAttentionKernel.Config(
            sliding_window_size=512,
            backend="triton",
        ).build()

        query = torch.randn(1, 128, 64, 512)
        local_kv = torch.randn(1, 1, 64, 512)
        sparse_kv = torch.randn(1, 1, 16, 512)
        kv_indices = torch.randint(0, 16, (1, 64, 8))
        attention_sink = torch.randn(128)
        expected = torch.randn(1, 128, 64, 512)

        with patch.object(
            attention_gym_module,
            "_selected_attention",
            return_value=expected,
        ) as mock_sa:
            result = kernel(query, local_kv, sparse_kv, kv_indices, attention_sink)

        self.assertIs(result, expected)
        self.assertEqual(mock_sa.call_args.kwargs["backend"], "triton")
        self.assertEqual(mock_sa.call_args.kwargs["sliding_window_size"], 512)
        self.assertIs(mock_sa.call_args.kwargs["attention_sink"], attention_sink)

    def test_kernel_cute_backend_disables_sink(self):
        kernel = SelectedAttentionKernel.Config(
            sliding_window_size=512,
            backend="cute",
        ).build()

        query = torch.randn(1, 128, 64, 512)
        local_kv = torch.randn(1, 1, 64, 512)
        sparse_kv = torch.randn(1, 1, 16, 512)
        kv_indices = torch.randint(0, 16, (1, 64, 8))
        attention_sink = torch.randn(128)
        expected = torch.randn(1, 128, 64, 512)

        with patch.object(
            attention_gym_module,
            "_selected_attention",
            return_value=expected,
        ) as mock_sa:
            result = kernel(query, local_kv, sparse_kv, kv_indices, attention_sink)

        self.assertIs(result, expected)
        self.assertEqual(mock_sa.call_args.kwargs["backend"], "cute")
        self.assertIsNone(mock_sa.call_args.kwargs["attention_sink"])

    def test_local_map_grad_placements_cover_every_tensor_input(self):
        sharding = _selected_attention_sharding_config()

        self.assertEqual(len(sharding.in_src_shardings), 5)
        self.assertEqual(len(sharding.local_map.in_grad_placements), 5)


if __name__ == "__main__":
    unittest.main()
