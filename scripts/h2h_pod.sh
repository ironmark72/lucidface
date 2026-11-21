#!/bin/bash
# Run the head-to-head on a RunPod pod, unattended-safe:
#   - hard cap: the pod stops itself after $CAP_HOURS whatever happens
#   - the pod stops itself when the job ends (success or failure)
#   - results are written to /root/bench/h2h.* (pull them with scp while it runs)
# Stopping (not terminating) keeps nothing billable except a volume disk, which this setup does not use.
# Prereqs: scripts/pod_setup.sh done; competitor repos/envs set (KEEP_DIR/KEEP_PY, PGT_DIR/PGT_PY,
# SVFR_DIR/SVFR_PY); Topaz exports (if any) uploaded to /root/bench/topaz/.
CAP_HOURS=${CAP_HOURS:-4}
B=${B:-/root/bench}
stop_pod() { echo "STOPPING POD $(date)"; runpodctl pod stop "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID"; }
( sleep $((CAP_HOURS * 3600)); echo "HARD CAP REACHED"; stop_pod ) &
trap stop_pod EXIT
set -x
cd /root/lucidface
export LUCIDFACE_CODEFORMER_DIR=${LUCIDFACE_CODEFORMER_DIR:-/root/models/CodeFormer}
python -u -m lucidface.cli bench download --dir "$B"
python -u -m lucidface.cli bench run --dir "$B" --levels "${LEVELS:-2}" --only "blurry,cf1.0+lock+steady.25"
echo PHASE lucidface_DONE
python -u scripts/headtohead.py "$B" --level "${LEVELS:-2}" --tools "${TOOLS:-keep,pgtformer,svfr,topaz}"
echo PHASE ALL_DONE
