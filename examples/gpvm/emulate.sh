#!/bin/bash
# Run table_job.sh on this machine the way a grid worker would: the tarball unpacked in a fresh folder, the worker
# image, and only /cvmfs plus the folders you name mounted. Anything the job still reads from a local area fails
# here, not on the grid. The lists in the tarball must then hold plain paths under <data mount>.
#
#   emulate.sh <tarball> <sample> <files per job> <work folder> <data mount> [<process number>]
set -euo pipefail
tar=$(realpath "${1:?tarball}") sample=${2:?sample} per=${3:?files per job} w=${4:?work folder} data=${5:?data mount} proc=${6:-0}
IMG=${IMG:-/cvmfs/singularity.opensciencegrid.org/fermilab/fnal-wn-el9:latest}
APPTAINER=${APPTAINER:-/cvmfs/oasis.opensciencegrid.org/mis/apptainer/current/bin/apptainer}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
rm -rf "$w"; mkdir -p "$w"/{tar,scratch,out}; w=$(realpath "$w")
tar -C "$w/tar" -xzf "$tar"; cp "$here/table_job.sh" "$w/tar/"
"$APPTAINER" exec --contain --cleanenv -B /cvmfs,"$data","$w" --pwd "$w/scratch" \
    --env INPUT_TAR_DIR_LOCAL="$w/tar",_CONDOR_SCRATCH_DIR="$w/scratch",PROCESS="$proc" \
    "$IMG" bash "$w/tar/table_job.sh" "$sample" "$per" "$w/out"
