# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses

from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder
from torchtitan.distributed.full_dtensor import (
    resolve_fsdp_mesh,
    resolve_sparse_fsdp_mesh,
    validate_config,
)
from torchtitan.distributed.pipeline_parallel import (
    _generate_llm_fqn_per_model_part,
    _get_pipeline_metadata,
    pipeline_llm,
)
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.models.deepseek_v4.model import DeepSeekV4Model


_HC_HEAD_FQNS = (
    "hc_head",
    "hc_head_fn",
    "hc_head_base",
    "hc_head_scale",
)


def parallelize_deepseek_v4(
    model: DeepSeekV4Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    if parallelism.spmd_backend == "full_dtensor":
        validate_config(parallel_dims, model)
        model.parallelize(parallel_dims)
    else:
        if parallel_dims.cp_enabled:
            raise NotImplementedError(
                "Context Parallel is not yet supported for DeepSeek V4 sparse attention."
            )
        if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
            model.parallelize(parallel_dims)

    if parallel_dims.tp_enabled:
        maybe_enable_async_tp(parallelism, compile_config, parallel_dims.get_mesh("tp"))

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)

    if model_compile_enabled:
        apply_compile(model, compile_config)

    if parallelism.spmd_backend == "full_dtensor":
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
        edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallel_dims)
    else:
        dp_mesh_names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)
        dp_mesh_dims = None
        edp_mesh = None
        edp_mesh_dims = None
        if parallel_dims.ep_enabled:
            edp_mesh_names = (
                ["dp_replicate", "efsdp"]
                if parallel_dims.dp_replicate_enabled
                else ["efsdp"]
            )
            edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
        dp_mesh_dims=dp_mesh_dims,
        edp_mesh_dims=edp_mesh_dims,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )

    return model


def _validate_deepseek_v4_pipeline_modules(
    model: DeepSeekV4Model,
    module_fqns_per_model_part: list[list[str]],
) -> None:
    if not module_fqns_per_model_part:
        raise ValueError(
            "DeepSeek V4 pipeline parallelism requires at least one model part."
        )

    required_first_stage_fqns = {"tok_embeddings"}
    required_first_stage_fqns.update(
        f"layers.{layer_id}"
        for layer_id, layer in model.layers.items()
        if layer.moe_enabled and layer.moe.router.hash
    )
    invalid_locations = {
        module_fqn: [
            stage_idx
            for stage_idx, stage_fqns in enumerate(module_fqns_per_model_part)
            if module_fqn in stage_fqns
        ]
        for module_fqn in sorted(required_first_stage_fqns)
    }
    invalid_locations = {
        module_fqn: stage_indices
        for module_fqn, stage_indices in invalid_locations.items()
        if stage_indices != [0]
    }
    if invalid_locations:
        raise ValueError(
            "DeepSeek V4 pipeline parallelism requires tok_embeddings and all "
            "hash-routed layers on stage 0 because raw input IDs are not "
            f"transported between pipeline stages. Invalid locations: "
            f"{invalid_locations}."
        )

    last_stage_idx = len(module_fqns_per_model_part) - 1
    invalid_hc_head_locations = {
        fqn: [
            stage_idx
            for stage_idx, stage_fqns in enumerate(module_fqns_per_model_part)
            if fqn in stage_fqns
        ]
        for fqn in _HC_HEAD_FQNS
    }
    invalid_hc_head_locations = {
        fqn: stage_indices
        for fqn, stage_indices in invalid_hc_head_locations.items()
        if stage_indices not in ([], [last_stage_idx])
    }
    if invalid_hc_head_locations:
        raise ValueError(
            "DeepSeek V4 pipeline parallelism requires the hc_head module and "
            "its direct parameters only on the last stage. Invalid locations: "
            f"{invalid_hc_head_locations}."
        )


def pipeline_deepseek_v4(
    model: DeepSeekV4Model,
    *,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    model_config: DeepSeekV4Model.Config,
    **kwargs,
):
    """Assign DeepSeek V4-specific modules to pipeline stages."""
    if parallelism.module_fqns_per_model_part is None:
        (
            num_virtual_stages,
            num_layers,
            input_weight,
            output_weight,
        ) = _get_pipeline_metadata(parallel_dims, parallelism, model_config)
        module_fqns_per_model_part = _generate_llm_fqn_per_model_part(
            num_virtual_stages,
            num_layers,
            input_weight,
            output_weight,
        )
    else:
        module_fqns_per_model_part = [
            list(stage_fqns) for stage_fqns in parallelism.module_fqns_per_model_part
        ]

    _validate_deepseek_v4_pipeline_modules(
        model,
        module_fqns_per_model_part,
    )
    for fqn in _HC_HEAD_FQNS:
        if fqn not in module_fqns_per_model_part[-1]:
            module_fqns_per_model_part[-1].append(fqn)

    parallelism = dataclasses.replace(
        parallelism,
        module_fqns_per_model_part=module_fqns_per_model_part,
    )
    return pipeline_llm(
        model,
        parallel_dims=parallel_dims,
        parallelism=parallelism,
        model_config=model_config,
        **kwargs,
    )
