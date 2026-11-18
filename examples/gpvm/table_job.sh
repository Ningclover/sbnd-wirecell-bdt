#!/bin/bash
# One FermiGrid job: build the spec_v1 tables for one slice of a sample's file list.
#
#   table_job.sh <sample> <files per job> <output folder or root:// URL> [extra builder options, e.g. --objs]
#
# The job tarball (make_bundle.sh) holds tables/, spec/, the Python packages (pydeps/) and lists/<sample>.txt with one
# merged CAF per line: an xrootd URL on the grid, a plain path when emulating. $PROCESS picks the slice. Every input
# is copied to the local scratch disk first, under its own parent folder name, so that --file-id-from dirname gives
# the production's unit id and the stored path is relative to the sample root.
set -uo pipefail
sample=${1:?sample} per=${2:?files per job} out=${3:?output}; shift 3; extra=("$@")
here=${INPUT_TAR_DIR_LOCAL:?no tarball}; work=${_CONDOR_SCRATCH_DIR:-$PWD}; p=${PROCESS:-0}
list=$here/lists/$sample.txt
n=$(grep -c . "$list"); a=$(( p * per )); b=$(( a + per < n ? a + per : n ))
tag=$(printf '%06d_%06d' "$a" "$b")
export PYTHONPATH=$here/pydeps PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
mkdir -p "$work/in" "$work/out"
t_start=$(date +%s); echo "job $p on $(hostname): $sample files [$a, $b) of $n; start $(date '+%F %T')"
: > "$work/local.txt"; : > "$work/out/copyfail_${sample}_$tag.txt"
sed -n "$(( a + 1 )),${b}p" "$list" | while read -r src; do
    [ -n "$src" ] || continue
    d=$work/in/$(basename "$(dirname "$src")"); mkdir -p "$d"; f=$d/$(basename "$src")
    if [[ $src == root://* ]]; then xrdcp --nopbar --retry 3 -f "$src" "$f"; else cp "$src" "$f"; fi   # the grid gives xrdcp its token
    if [ -s "$f" ]; then echo "$f" >> "$work/local.txt"; else echo "$src" >> "$work/out/copyfail_${sample}_$tag.txt"; fi
done
t_copied=$(date +%s); in_bytes=$(du -sb "$work/in" | cut -f1)
python3 "$here/tables/build_tables.py" "$sample" "$work/out" --list "$work/local.txt" \
    --file-id-from dirname --root "$work/in" --tag "$tag" "${extra[@]}" || exit 1
t_built=$(date +%s)
# time record of this job, shipped with the tables: seconds for copy-in and build, bytes copied, where it ran
cat > "$work/out/timing_${sample}_$tag.json" <<TIMING
{"sample": "$sample", "slice": "$tag", "process": $p, "cluster": "${CLUSTER:-}", "host": "$(hostname)", "cpu_model": "$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')",
 "start": $t_start, "files_copied": $(grep -c . "$work/local.txt"), "bytes_copied": $in_bytes, "copy_in_s": $(( t_copied - t_start )), "build_s": $(( t_built - t_copied ))}
TIMING
for f in "$work"/out/*; do
    if [[ $out == root://* ]]; then xrdcp --nopbar --retry 3 -f -p "$f" "$out/$(basename "$f")" || exit 2; else mkdir -p "$out" && cp "$f" "$out/" || exit 2; fi
done
t_end=$(date +%s)
echo "done: $(ls "$work/out" | wc -l) files -> $out; copy-in $(( t_copied - t_start )) s, build $(( t_built - t_copied )) s, copy-out $(( t_end - t_built )) s, total $(( t_end - t_start )) s; end $(date '+%F %T')"
