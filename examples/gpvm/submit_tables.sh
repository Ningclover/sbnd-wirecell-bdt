#!/bin/bash
# Submit table_job.sh over one sample's list with jobsub_lite, one job per slice.
#
#   submit_tables.sh <tarball> <sample> <files per job> <number of files in lists/<sample>.txt> <output root:// URL>
#
# Set GROUP to your experiment. Needs a token (htgettoken) and a login shell (jobsub_lite's environment).
# --dry-run as a sixth argument prints the command only.
set -euo pipefail
tar=$(realpath "${1:?tarball}") sample=${2:?sample} per=${3:?files per job} n=${4:?number of files} out=${5:?output URL} dry=${6:-}
GROUP=${GROUP:?set GROUP to your experiment, e.g. export GROUP=<experiment>}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
njobs=$(( (n + per - 1) / per ))
cmd=(jobsub_submit -G "$GROUP" -N "$njobs" --memory="${MEMORY:-2000MB}" --disk="${DISK:-6GB}" --expected-lifetime="${LIFETIME:-2h}"
     --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC
     --singularity-image "${IMG:-/cvmfs/singularity.opensciencegrid.org/fermilab/fnal-wn-el9:latest}"
     --tar_file_name "dropbox://$tar" "file://$here/table_job.sh" "$sample" "$per" "$out")
echo "$sample: $n files, $per per job -> $njobs jobs, output $out"; printf '%q ' "${cmd[@]}"; echo
[ "$dry" = "--dry-run" ] || "${cmd[@]}"
