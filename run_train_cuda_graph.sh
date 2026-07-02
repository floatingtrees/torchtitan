#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export TORCHINDUCTOR_CUDAGRAPHS="${TORCHINDUCTOR_CUDAGRAPHS:-1}"
export TORCHINDUCTOR_CUDAGRAPH_OR_ERROR="${TORCHINDUCTOR_CUDAGRAPH_OR_ERROR:-1}"
export TORCHINDUCTOR_GRAPH_PARTITION="${TORCHINDUCTOR_GRAPH_PARTITION:-0}"

cd "$repo_root"
exec ./run_train.sh \
    --compile.enable \
    --compile.backend inductor \
    "$@"
