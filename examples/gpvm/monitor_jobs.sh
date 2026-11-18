#!/bin/bash
# Follow your grid jobs until none is left in the queue and keep a time record. Run it inside tmux or screen, so it
# survives a closed terminal. Writes into <monitor folder>:
#   queue_snapshots.log   every poll: time, jobs idle / running / held, slices arrived per sample folder
#   job_states.tsv        one line per state change of a job: epoch, time, job id, state (I, R, H, GONE)
#   MONITOR_DONE          written when the queue has been empty for two polls
#
#   monitor_jobs.sh <group> <output folder as mounted, holding one folder per sample> <monitor folder> [poll seconds]
#
# Start it right after submitting. It follows every job of yours in that group, so do not run other jobs meanwhile.
set -u
group=${1:?group}; outroot=${2:?output folder}; M=${3:?monitor folder}; poll=${4:-60}
mkdir -p "$M"; declare -A last; seen_any=0; empty_polls=0
echo "monitor started $(date '+%F %T')" | tee -a "$M/queue_snapshots.log"
while true; do
    now=$(date +%s); stamp=$(date '+%F %T')
    q=$(jobsub_q -G "$group" --user "$USER" 2>/dev/null | awk '$1 ~ /@/ {print $1, $6}')
    declare -A cur=(); ni=0; nr=0; nh=0
    while read -r id st; do
        [ -n "$id" ] || continue; cur[$id]=$st; seen_any=1
        case $st in I) ni=$((ni+1));; R) nr=$((nr+1));; H) nh=$((nh+1));; esac
        if [ "${last[$id]:-}" != "$st" ]; then printf '%s\t%s\t%s\t%s\n' "$now" "$stamp" "$id" "$st" >> "$M/job_states.tsv"; last[$id]=$st; fi
    done <<< "$q"
    for id in "${!last[@]}"; do
        if [ -z "${cur[$id]:-}" ] && [ "${last[$id]}" != GONE ]; then printf '%s\t%s\t%s\tGONE\n' "$now" "$stamp" "$id" >> "$M/job_states.tsv"; last[$id]=GONE; fi
    done
    arrived=""
    for d in "$outroot"/*/; do [ -d "$d" ] && arrived+=" $(basename "$d") $(timeout 60 ls "$d" 2>/dev/null | grep -c '^build_')"; done
    echo "$stamp idle $ni running $nr held $nh | slices arrived:${arrived:- none}" | tee -a "$M/queue_snapshots.log"
    [ "$nh" -gt 0 ] && echo "$stamp WARNING: $nh held job(s); look with jobsub_q --hold -G $group --user $USER" | tee -a "$M/queue_snapshots.log"
    if [ $(( ni + nr + nh )) -eq 0 ]; then empty_polls=$((empty_polls+1)); else empty_polls=0; fi
    # finished: jobs were seen and the queue has been empty for two polls; give up if nothing ever appears in 30 polls
    if { [ "$seen_any" = 1 ] && [ "$empty_polls" -ge 2 ]; } || [ "$empty_polls" -ge 30 ]; then break; fi
    sleep "$poll"
done
echo "queue empty $(date '+%F %T')" | tee -a "$M/queue_snapshots.log"; date '+%F %T' > "$M/MONITOR_DONE"
