#!/bin/bash
# Submit table_job.sh over one sample's list with jobsub_lite, one job per slice.
#
#   submit_tables.sh <tarball> <sample> <files per job> <number of files in lists/<sample>.txt> <output root:// URL>
#
# BUILD_ARGS holds extra builder options for every job, e.g. BUILD_ARGS=--objs for the per-object tables.
#
# Set GROUP to your experiment. Needs a token (htgettoken) and a login shell (jobsub_lite's environment).
# --dry-run as a sixth argument prints the command only. The first real submission writes submit_time.txt in the
# current folder (SUBMIT_TIME_FILE to change), which timing_summary.py reads.
set -euo pipefail
tar=$(realpath "${1:?tarball}") sample=${2:?sample} per=${3:?files per job} n=${4:?number of files} out=${5:?output URL} dry=${6:-}
GROUP=${GROUP:?set GROUP to your experiment, e.g. export GROUP=<experiment>}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
njobs=$(( (n + per - 1) / per ))
cmd=(jobsub_submit -G "$GROUP" --role=Analysis -N "$njobs" --cpu=1 --memory="${MEMORY:-2000MB}" --disk="${DISK:-6GB}"
     --expected-lifetime="${LIFETIME:-2h}" --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC
     --singularity-image "${IMG:-/cvmfs/singularity.opensciencegrid.org/fermilab/fnal-wn-el9:latest}"
     --append_condor_requirements='(TARGET.HAS_SINGULARITY=?=true)' --mail_never
     --tar_file_name "dropbox://$tar" "file://$here/table_job.sh" "$sample" "$per" "$out" ${BUILD_ARGS:-})
echo "$sample: $n files, $per per job -> $njobs jobs, output $out"; printf '%q ' "${cmd[@]}"; echo
[ "$dry" = "--dry-run" ] && exit 0
# the submit time, for timing_summary.py: epoch first, then the readable time
[ -f "${SUBMIT_TIME_FILE:-submit_time.txt}" ] || { date +%s; date '+%F %T'; } > "${SUBMIT_TIME_FILE:-submit_time.txt}"
"${cmd[@]}"
