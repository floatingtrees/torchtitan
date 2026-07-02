#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
remote="${1:-origin}"
branches=(
    dev/custom_kernel_cuda_graph
    dev/allgather_cuda_graph
)

git -C "$repo_root" remote get-url "$remote" >/dev/null
git -C "$repo_root" push --set-upstream "$remote" "${branches[@]}"
