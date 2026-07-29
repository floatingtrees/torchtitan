# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch

import torchtitan.models.deepseek_v4.attention_gym as attention_gym_module
import torchtitan.models.deepseek_v4.attention_gym_csa as attention_gym_csa_module
from torchtitan.models.deepseek_v4.attention_gym import (
    _hca_sharding_config,
    _swa_sharding_config,
    attention_gym,
    AttentionGymAttention,
    AttentionGymHCAKernel,
    AttentionGymSWAKernel,
)
from torchtitan.models.deepseek_v4.attention_gym_csa import AttentionGymCSAKernel
from torchtitan.models.deepseek_v4.config_registry import deepseek_large_debug_config


class TestDeepSeekV4AttentionGymOverride(unittest.TestCase):
    def test_large_debug_config_exercises_all_attention_types(self):
        model_config = deepseek_large_debug_config().model_spec.model

        self.assertEqual(model_config.compress_ratios, (1, 4, 128, 4))
        self.assertEqual(
            [layer.attention.n_heads for layer in model_config.layers],
            [128, 128, 128, 128],
        )

    def test_factory_adds_all_kernel_configs(self):
        attention_config = (
            deepseek_large_debug_config().model_spec.model.layers[2].attention
        )

        with patch.object(
            attention_gym_module,
            "_ATTENTION_GYM_IMPORT_ERROR",
            None,
        ):
            replacement = attention_gym(attention_config)

        self.assertIsInstance(replacement, AttentionGymAttention.Config)
        self.assertEqual(replacement.csa_kernel.compression_ratio, 128)
        self.assertEqual(replacement.hca_kernel.compression_ratio, 128)
        self.assertEqual(
            replacement.swa_kernel.sliding_window_size,
            attention_config.window_size,
        )

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
            self.assertRaisesRegex(ImportError, "floatingtrees/attention-gym"),
        ):
            attention_gym(attention_config)

    def test_hca_kernel_uses_cute_backend(self):
        kernel = AttentionGymHCAKernel.Config(
            compression_ratio=128,
            sliding_window_size=16,
            rope_dims=32,
        ).build()
        inputs = [torch.randn(1) for _ in range(8)]
        expected = torch.randn(1)

        with patch.object(
            attention_gym_module,
            "heavily_compressed_attention",
            return_value=expected,
        ) as hca:
            result = kernel(*inputs)

        self.assertIs(result, expected)
        self.assertEqual(hca.call_args.args[-4:], (128, 16, 32, True))
        self.assertEqual(hca.call_args.kwargs, {"backend": "cute"})

    def test_csa_kernel_uses_preloaded_cute_backend(self):
        kernel = AttentionGymCSAKernel.Config(
            compression_ratio=4,
            num_topk_blocks=64,
            sliding_window_size=512,
            rope_dims=64,
        ).build()
        inputs = [torch.randn(1) for _ in range(20)]
        expected = torch.randn(1)

        with patch.object(
            attention_gym_csa_module,
            "_compressed_sparse_attention_cute",
            return_value=expected,
        ) as csa:
            result = kernel(*inputs)

        self.assertIs(result, expected)
        self.assertEqual(csa.call_args.args[-5:], (4, 64, 512, 64, True))
        self.assertEqual(csa.call_args.kwargs, {})

    def test_swa_kernel_uses_cute_backend(self):
        kernel = AttentionGymSWAKernel.Config(
            sliding_window_size=16,
            rope_dims=32,
        ).build()
        inputs = [torch.randn(1) for _ in range(4)]
        expected = torch.randn(1)

        with patch.object(
            attention_gym_module,
            "sliding_window_attention",
            return_value=expected,
        ) as swa:
            result = kernel(*inputs)

        self.assertIs(result, expected)
        self.assertEqual(swa.call_args.args[-3:], (16, 32, True))
        self.assertEqual(swa.call_args.kwargs, {"backend": "cute"})

    def test_local_map_grad_placements_cover_every_tensor_input(self):
        hca_sharding = _hca_sharding_config()
        swa_sharding = _swa_sharding_config()

        self.assertEqual(len(hca_sharding.in_src_shardings), 8)
        self.assertEqual(
            len(hca_sharding.local_map.in_grad_placements),
            8,
        )
        self.assertEqual(len(swa_sharding.in_src_shardings), 4)
        self.assertEqual(
            len(swa_sharding.local_map.in_grad_placements),
            4,
        )


if __name__ == "__main__":
    unittest.main()
