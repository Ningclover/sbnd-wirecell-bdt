#!/bin/bash
# Build the job tarball: this repository's tables/, spec/ and evaluation/, the Python packages the builder needs
# (installed with pip into pydeps/, used through PYTHONPATH), and your file lists.
#
#   make_bundle.sh <folder with one <sample>.txt per sample> <build folder> <tarball>
#
# The worker image has the same OS and Python 3.9 as the gpvm, so packages installed here run there. Set PYDEPS to
# an existing pydeps folder to reuse it instead of installing again.
set -euo pipefail
lists=${1:?lists folder} build=${2:?build folder} tar=${3:?tarball}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
rm -rf "$build/pkg"; mkdir -p "$build/pkg/lists"
if [ -n "${PYDEPS:-}" ]; then cp -r "$PYDEPS" "$build/pkg/pydeps"
else python3 -m pip install --quiet --no-cache-dir --target "$build/pkg/pydeps" numpy pandas pyarrow uproot awkward; fi
cp -r "$repo/tables" "$repo/spec" "$repo/evaluation" "$build/pkg/"
cp "$lists"/*.txt "$build/pkg/lists/"
( cd "$build/pkg" && git -C "$repo" rev-parse HEAD > REPO_COMMIT.txt && python3 -c "import sys; print(sys.version)" > BUILD_PYTHON.txt )
tar -C "$build/pkg" --exclude='__pycache__' -czf "$tar" .
echo "wrote $tar ($(du -h "$tar" | cut -f1)); lists: $(ls "$build/pkg/lists" | tr '\n' ' ')"
