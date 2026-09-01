# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch
from attn_gym.sparse.selected_attention import selected_attention

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.models.deepseek_v4 import _get_indexer_loss_scale, deepseek_v4_configs
from torchtitan.models.deepseek_v4.attention import (
    _indexer_loss,
    _InjectAuxLoss,
    CompressedSparseAttention,
)
from torchtitan.models.deepseek_v4.compressor import Indexer


def _reference_indexer_loss(query, compressed_kv, lse, indexer_logits, valid):
    scale = query.shape[-1] ** -0.5
    eps = torch.finfo(torch.float32).tiny
    losses = []
    for batch_idx in range(query.shape[0]):
        for token_idx in range(query.shape[2]):
            row_valid = valid[batch_idx, token_idx]
            if not row_valid.any():
                continue
            masses = []
            for selected_idx in range(compressed_kv.shape[2]):
                mass = query.new_zeros(())
                if row_valid[selected_idx]:
                    for head_idx in range(query.shape[1]):
                        score = torch.dot(
                            query[batch_idx, head_idx, token_idx].float(),
                            compressed_kv[batch_idx, token_idx, selected_idx].float(),
                        )
                        mass = mass + torch.exp(
                            score * scale - lse[batch_idx, head_idx, token_idx].float()
                        )
                masses.append(mass)
            masses = torch.stack(masses)
            teacher = masses / masses.sum().clamp_min(eps)
            student_log_probs = torch.log_softmax(
                indexer_logits[batch_idx, token_idx, row_valid].float(), dim=0
            )
            teacher_valid = teacher[row_valid]
            losses.append(
                (
                    teacher_valid
                    * (teacher_valid.clamp_min(eps).log() - student_log_probs)
                ).sum()
            )
    return torch.stack(losses).mean()


class TestDeepSeekV4SelectedAttention(unittest.TestCase):
    def test_indexer_select_returns_indices_and_gathered_logits(self):
        torch.manual_seed(0)
        seqlen, ratio, topk = 12, 3, 3
        idx_q = torch.randn(seqlen, 2, 4)
        idx_k = torch.randn(seqlen // ratio, 4)
        idx_w = torch.randn(seqlen, 2)

        indices, selected_logits = Indexer.select(
            idx_q,
            idx_k,
            idx_w,
            seqlen=seqlen,
            ratio=ratio,
            topk=topk,
        )

        scores = torch.einsum("shd,td->sht", idx_q, idx_k)
        scores = torch.relu(scores) * idx_w.unsqueeze(-1)
        scores = scores.sum(dim=1)
        causal_limit = torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        causal_mask = torch.arange(seqlen // ratio).expand(seqlen, -1) >= causal_limit
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        expected_logits, expected_indices = scores.topk(topk, dim=-1)

        self.assertTrue(torch.equal(indices, expected_indices))
        torch.testing.assert_close(selected_logits, expected_logits, rtol=0, atol=0)
        torch.testing.assert_close(
            selected_logits,
            scores.gather(-1, indices),
            rtol=0,
            atol=0,
        )

    def test_indexer_loss_value_and_gradient_contract(self):
        torch.manual_seed(1)
        query = torch.randn(1, 2, 3, 4, requires_grad=True)
        compressed_kv = torch.randn(1, 3, 3, 4, requires_grad=True)
        lse = (torch.randn(1, 2, 3) + 3).requires_grad_()
        indexer_logits = torch.randn(1, 3, 3, requires_grad=True)
        valid = torch.tensor(
            [[[False, False, False], [True, True, False], [True, True, True]]]
        )
        reference_logits = indexer_logits.detach().clone().requires_grad_()

        actual = _indexer_loss(query, compressed_kv, lse, indexer_logits, valid)
        expected = _reference_indexer_loss(
            query.detach(),
            compressed_kv.detach(),
            lse.detach(),
            reference_logits,
            valid,
        )
        actual.backward()
        expected.backward()

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            indexer_logits.grad, reference_logits.grad, rtol=1e-5, atol=1e-6
        )
        self.assertIsNone(query.grad)
        self.assertIsNone(compressed_kv.grad)
        self.assertIsNone(lse.grad)
        indexer_grad = indexer_logits.grad
        assert indexer_grad is not None
        self.assertGreater(indexer_grad.norm().item(), 0)
        self.assertTrue(torch.equal(indexer_grad[~valid], torch.zeros(4)))

    def test_aux_loss_injection_matches_explicit_objective(self):
        torch.manual_seed(2)
        scale = 0.125
        carrier = torch.randn(3, 4, requires_grad=True)
        aux_loss = torch.randn((), requires_grad=True)
        reference_carrier = carrier.detach().clone().requires_grad_()
        reference_aux_loss = aux_loss.detach().clone().requires_grad_()

        result = _InjectAuxLoss.apply(carrier, aux_loss, scale)
        torch.testing.assert_close(result, carrier, rtol=0, atol=0)
        result.square().sum().backward()
        (reference_carrier.square().sum() + scale * reference_aux_loss).backward()

        torch.testing.assert_close(carrier.grad, reference_carrier.grad, rtol=0, atol=0)
        torch.testing.assert_close(
            aux_loss.grad, reference_aux_loss.grad, rtol=0, atol=0
        )

    def test_indexer_loss_scale_covers_dp_ga_and_pp(self):
        training = TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=64,
            num_tokens_per_train_step=1024,
        )
        parallel_dims = ParallelDims(
            dp_replicate=2,
            dp_shard=2,
            cp=1,
            tp=1,
            pp=2,
            ep=1,
            world_size=8,
        )
        parallelism = ParallelismConfig(
            data_parallel_replicate_degree=2,
            data_parallel_shard_degree=2,
            pipeline_parallel_degree=2,
            num_pp_microbatches=2,
        )

        self.assertEqual(
            _get_indexer_loss_scale(training, parallel_dims, parallelism),
            1 / 16,
        )

        training = TrainingConfig(num_tokens_per_microbatch_per_dp_rank=64)
        self.assertEqual(
            _get_indexer_loss_scale(training, parallel_dims, parallelism),
            1 / 8,
        )

        parallel_dims = ParallelDims(
            dp_replicate=2,
            dp_shard=2,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=4,
        )
        self.assertEqual(
            _get_indexer_loss_scale(training, parallel_dims, parallelism),
            1 / 4,
        )

    def test_indexer_loss_uses_global_tp_teacher(self):
        torch.manual_seed(3)
        query = torch.randn(1, 2, 2, 4)
        compressed_kv = torch.randn(1, 2, 3, 4)
        lse = torch.randn(1, 2, 2) + 4
        valid = torch.tensor([[[True, True, False], [True, True, True]]])
        logits = torch.randn(1, 2, 3, requires_grad=True)
        reference_logits = logits.detach().clone().requires_grad_()
        scale = query.shape[-1] ** -0.5
        remote_mass = torch.exp(
            torch.einsum(
                "bntd,btkd->bntk",
                query[:, 1:].float(),
                compressed_kv.float(),
            )
            * scale
            - lse[:, 1:].float()[..., None]
        ).sum(dim=1)
        remote_mass = remote_mass * valid.float()

        tp_group = object()
        with patch(
            "torchtitan.models.deepseek_v4.attention.dist_sum_tensor",
            side_effect=lambda local_mass, **_: local_mass + remote_mass,
        ) as reduce_mock:
            actual = _indexer_loss(
                query[:, :1],
                compressed_kv,
                lse[:, :1],
                logits,
                valid,
                tp_group=tp_group,
            )
        expected = _reference_indexer_loss(
            query,
            compressed_kv,
            lse,
            reference_logits,
            valid,
        )
        actual.backward()
        expected.backward()

        reduce_mock.assert_called_once()
        self.assertIs(reduce_mock.call_args.kwargs["extra_pg"], tp_group)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            logits.grad,
            reference_logits.grad,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_indexer_loss_log_softmax_is_stable(self):
        query = torch.zeros(1, 2, 1, 4)
        compressed_kv = torch.zeros(1, 1, 2, 4)
        lse = torch.zeros(1, 2, 1)
        valid = torch.ones(1, 1, 2, dtype=torch.bool)
        logits = torch.tensor([[[1000.0, -1000.0]]], requires_grad=True)
        reference_logits = logits.detach().clone().requires_grad_()

        actual = _indexer_loss(query, compressed_kv, lse, logits, valid)
        expected = _reference_indexer_loss(
            query,
            compressed_kv,
            lse,
            reference_logits,
            valid,
        )
        actual.backward()
        expected.backward()

        self.assertTrue(torch.isfinite(actual))
        self.assertTrue(torch.isfinite(logits.grad).all())
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            logits.grad,
            reference_logits.grad,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_eager_csa_matches_selected_attention_forward_and_backward(self):
        torch.manual_seed(3)
        seqlen, num_heads, head_dim = 8, 2, 4
        ratio, topk, window_size = 2, 2, 3
        num_compressed = seqlen // ratio
        module = CompressedSparseAttention.Config(
            window_size=window_size,
            compress_ratio=ratio,
            softmax_scale=head_dim**-0.5,
            index_topk=topk,
            backend="eager",
        ).build()

        q = torch.randn(seqlen, num_heads, head_dim, requires_grad=True)
        local_kv = torch.randn(seqlen, head_dim, requires_grad=True)
        compressed_kv = torch.randn(num_compressed, head_dim, requires_grad=True)
        idx_q = (torch.rand(seqlen, 3, 4) + 0.2).requires_grad_()
        idx_k = (torch.rand(num_compressed, 4) + 0.2).requires_grad_()
        idx_w = (torch.rand(seqlen, 3) + 0.2).requires_grad_()
        sink = torch.randn(num_heads, requires_grad=True)
        reference_q = q.detach().clone().requires_grad_()
        reference_local_kv = local_kv.detach().clone().requires_grad_()
        reference_compressed_kv = compressed_kv.detach().clone().requires_grad_()
        reference_sink = sink.detach().clone().requires_grad_()

        output = module(
            q,
            local_kv,
            compressed_kv,
            idx_q,
            idx_k,
            idx_w,
            sink,
        )
        raw_indices, _ = Indexer.select(
            idx_q.detach(),
            idx_k.detach(),
            idx_w.detach(),
            seqlen=seqlen,
            ratio=ratio,
            topk=topk,
        )
        causal_limit = torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        causal_indices = torch.where(raw_indices < causal_limit, raw_indices, -1)
        reference_output = (
            selected_attention(
                reference_q.transpose(0, 1).unsqueeze(0),
                reference_local_kv.unsqueeze(0).unsqueeze(0),
                reference_compressed_kv.unsqueeze(0).unsqueeze(0),
                causal_indices.unsqueeze(0),
                reference_sink,
                None,
                window_size,
                backend="eager",
            )
            .squeeze(0)
            .transpose(0, 1)
        )

        torch.testing.assert_close(output, reference_output, rtol=0, atol=0)
        output_weight = torch.randn_like(output)
        (output * output_weight).sum().backward()
        (reference_output * output_weight).sum().backward()

        for actual, expected in (
            (q, reference_q),
            (local_kv, reference_local_kv),
            (compressed_kv, reference_compressed_kv),
            (sink, reference_sink),
        ):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-6)
        for feature in (idx_q, idx_k, idx_w):
            feature_grad = feature.grad
            assert feature_grad is not None
            self.assertGreater(feature_grad.norm().item(), 0)
            self.assertTrue(torch.isfinite(feature_grad).all())

    def test_csa_skips_indexer_loss_without_training_gradients(self):
        torch.manual_seed(4)
        seqlen, head_dim = 8, 4
        module = CompressedSparseAttention.Config(
            window_size=3,
            compress_ratio=2,
            softmax_scale=head_dim**-0.5,
            index_topk=2,
            backend="eager",
        ).build()
        args = (
            torch.randn(seqlen, 2, head_dim),
            torch.randn(seqlen, head_dim),
            torch.randn(seqlen // 2, head_dim),
            torch.randn(seqlen, 3, 4),
            torch.randn(seqlen // 2, 4),
            torch.randn(seqlen, 3),
            torch.randn(2),
        )

        for training, grad_enabled in ((False, True), (True, False)):
            module.train(training)
            grad_context = torch.enable_grad() if grad_enabled else torch.no_grad()
            with (
                grad_context,
                patch.object(
                    module,
                    "_forward_selected",
                    wraps=module._forward_selected,
                ) as selected_mock,
                patch(
                    "torchtitan.models.deepseek_v4.attention._indexer_loss"
                ) as loss_mock,
            ):
                module(*args)
            loss_mock.assert_not_called()
            self.assertFalse(selected_mock.call_args.kwargs["return_lse"])

    def test_model_configs_select_backend_by_head_dim(self):
        debug_config = deepseek_v4_configs["debugmodel"]()
        flash_config = deepseek_v4_configs["deepseek_v4_flash"]()
        debug_backends = [
            layer.attention.inner_attention.backend
            for layer in debug_config.layers
            if layer.attention.compress_ratio > 1
        ]
        flash_backends = [
            layer.attention.inner_attention.backend
            for layer in flash_config.layers
            if layer.attention.compress_ratio > 1
        ]

        self.assertEqual(debug_backends, ["triton", "triton"])
        self.assertTrue(flash_backends)
        self.assertEqual(set(flash_backends), {"cute"})


if __name__ == "__main__":
    unittest.main()
