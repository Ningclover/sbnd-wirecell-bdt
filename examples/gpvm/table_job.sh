#!/bin/bash
# One FermiGrid job: build the spec_v1 tables for one slice of a sample's file list.
#
#   table_job.sh <sample> <files per job> <output folder or root:// URL>
#
# The job tarball (make_bundle.sh) holds tables/, spec/, the Python packages (pydeps/) and lists/<sample>.txt with one
# merged CAF per line: an xrootd URL on the grid, a plain path when emulating. $PROCESS picks the slice. Every input
# is copied to the local scratch disk first, under its own parent folder name, so that --file-id-from dirname gives
# the production's unit id and the stored path is relative to the sample root.
set -uo pipefail
sample=${1:?sample} per=${2:?files per job} out=${3:?output}
here=${INPUT_TAR_DIR_LOCAL:?no tarball}; work=${_CONDOR_SCRATCH_DIR:-$PWD}; p=${PROCESS:-0}
list=$here/lists/$sample.txt
n=$(grep -c . "$list"); a=$(( p * per )); b=$(( a + per < n ? a + per : n ))
tag=$(printf '%06d_%06d' "$a" "$b")
export PYTHONPATH=$here/pydeps PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
mkdir -p "$work/in" "$work/out"
echo "job $p on $(hostname): $sample files [$a, $b) of $n"
: > "$work/local.txt"; : > "$work/out/copyfail_${sample}_$tag.txt"
sed -n "$(( a + 1 )),${b}p" "$list" | while read -r src; do
    [ -n "$src" ] || continue
    d=$work/in/$(basename "$(dirname "$src")"); mkdir -p "$d"; f=$d/$(basename "$src")
    if [[ $src == root://* ]]; then xrdcp -s -f "$src" "$f"; else cp "$src" "$f"; fi   # the grid gives xrdcp its token
    if [ -s "$f" ]; then echo "$f" >> "$work/local.txt"; else echo "$src" >> "$work/out/copyfail_${sample}_$tag.txt"; fi
done
python3 "$here/tables/build_tables.py" "$sample" "$work/out" --list "$work/local.txt" \
    --file-id-from dirname --root "$work/in" --tag "$tag" || exit 1
for f in "$work"/out/*; do
    if [[ $out == root://* ]]; then xrdcp -s "$f" "$out/$(basename "$f")" || exit 2; else mkdir -p "$out" && cp "$f" "$out/" || exit 2; fi
done
echo "done: $(ls "$work/out" | wc -l) files -> $out"
