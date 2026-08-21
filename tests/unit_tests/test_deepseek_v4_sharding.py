# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtitan.models.common.decoder_sharding import norm_config
from torchtitan.models.deepseek_v4 import deepseek_v4_configs
from torchtitan.models.deepseek_v4.sharding import set_deepseek_v4_layer_sharding


class TestDeepSeekV4Sharding(unittest.TestCase):
    def test_block_norms_follow_sequence_parallel_layout(self):
        for enable_sp in (False, True):
            with self.subTest(enable_sp=enable_sp):
                layer_config = deepseek_v4_configs["debugmodel"]().layers[0]
                set_deepseek_v4_layer_sharding(
                    layer_config,
                    enable_sp=enable_sp,
                    enable_ep=False,
                )

                expected = norm_config(enable_sp=enable_sp)
                self.assertEqual(
                    layer_config.attention_norm.sharding_config,
                    expected,
                )
                self.assertEqual(
                    layer_config.ffn_norm.sharding_config,
                    expected,
                )

                internal_norm = norm_config(enable_sp=False)
                self.assertEqual(
                    layer_config.attention.q_norm.sharding_config,
                    internal_norm,
                )
                self.assertEqual(
                    layer_config.attention.kv_norm.sharding_config,
                    internal_norm,
                )


if __name__ == "__main__":
    unittest.main()
