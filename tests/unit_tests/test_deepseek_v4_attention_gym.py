# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

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
from torchtitan.models.deepseek_v4.attention_gym_csa import (
    _make_rope_tables_capture_safe,
    AttentionGymCSAKernel,
)
from torchtitan.models.deepseek_v4.config_registry import deepseek_large_debug_config
from torchtitan.models.deepseek_v4.model import DeepSeekV4Model


class TestDeepSeekV4AttentionGymOverride(unittest.TestCase):
    def test_large_debug_config_exercises_all_attention_types(self):
        config = deepseek_large_debug_config()
        assert config.model_spec is not None
        model_config = cast(DeepSeekV4Model.Config, config.model_spec.model)

        self.assertEqual(model_config.compress_ratios, (1, 4, 128, 4) * 2)
        self.assertEqual(
            [layer.attention.n_heads for layer in model_config.layers],
            [128] * 8,
        )

    def test_factory_adds_all_kernel_configs(self):
        config = deepseek_large_debug_config()
        assert config.model_spec is not None
        model_config = cast(DeepSeekV4Model.Config, config.model_spec.model)
        attention_config = model_config.layers[2].attention

        with patch.object(
            attention_gym_module,
            "_ATTENTION_GYM_IMPORT_ERROR",
            None,
        ):
            replacement = attention_gym(attention_config)

        self.assertIsInstance(replacement, AttentionGymAttention.Config)
        assert replacement.csa_kernel is not None
        assert replacement.hca_kernel is not None
        assert replacement.swa_kernel is not None
        self.assertEqual(replacement.csa_kernel.compression_ratio, 128)
        self.assertEqual(replacement.hca_kernel.compression_ratio, 128)
        self.assertEqual(
            replacement.swa_kernel.sliding_window_size,
            attention_config.window_size,
        )

    def test_factory_requires_attention_gym(self):
        config = deepseek_large_debug_config()
        assert config.model_spec is not None
        model_config = cast(DeepSeekV4Model.Config, config.model_spec.model)
        attention_config = model_config.layers[0].attention
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
        assert hca_sharding.in_src_shardings is not None
        assert hca_sharding.local_map is not None
        assert swa_sharding.in_src_shardings is not None
        assert swa_sharding.local_map is not None
        assert hca_sharding.local_map.in_grad_placements is not None
        assert swa_sharding.local_map.in_grad_placements is not None

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

    def test_rope_tables_reuse_warmup_stream_during_capture(self):
        key = (0, 1024, 64)
        cos = Mock()
        sin = Mock()
        original = Mock(return_value=(cos, sin))
        stream = Mock()
        module = SimpleNamespace(
            _rope_tables=original,
            _rope_table_cache={key: (cos, sin, Mock())},
        )
        _make_rope_tables_capture_safe(module)

        with (
            patch("torch.cuda.current_stream", return_value=stream),
            patch("torch.cuda.is_current_stream_capturing", side_effect=(False, True)),
        ):
            self.assertEqual(module._rope_tables(*key), (cos, sin))
            self.assertEqual(module._rope_tables(*key), (cos, sin))

        original.assert_called_once_with(*key)
        cos.record_stream.assert_called_once_with(stream)
        sin.record_stream.assert_called_once_with(stream)

    def test_rope_tables_require_warmup_on_capture_stream(self):
        key = (0, 1024, 64)
        module = SimpleNamespace(
            _rope_tables=Mock(),
            _rope_table_cache={key: (Mock(), Mock(), Mock())},
        )
        _make_rope_tables_capture_safe(module)

        with (
            patch("torch.cuda.current_stream", return_value=Mock()),
            patch("torch.cuda.is_current_stream_capturing", return_value=True),
            self.assertRaisesRegex(RuntimeError, "must be warmed up"),
        ):
            module._rope_tables(*key)


if __name__ == "__main__":
    unittest.main()
