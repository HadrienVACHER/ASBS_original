#!/bin/bash
# Vast.ai instance, rules/rules_smc_noise.md (amendment ter). Env: STOP_AT = unix time after which no computation runs (cost cap). Stages in priority order; a stage is not started (nor the later ones)
# if its projected duration (from the measured throughput of the first B run) exceeds the remaining time. Runs already finished (JSON present) are skipped.
set -u
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
cd /workspace/asbs; mkdir -p logs figures/smc_noise results/smc_noise
P=/workspace/progress.txt; : > $P; echo running > /workspace/status.txt
say() { echo "$(date +%s) $*" >> $P; }
left() { echo $(( STOP_AT - $(date +%s) )); }
one() { out=$1; shift; tag=$(echo "$*" | tr ' ' '_'); [ -f "$out" ] && return 0; L_=$(left); [ $L_ -le 20 ] && { say "SKIP_deadline $tag"; return 1; }
  say "start $tag"; timeout $L_ python scripts/smc_noise.py "$@" > logs/smc_$tag.log 2>&1; rc=$?; say "end $tag rc=$rc"; return $rc; }
# 1) smoke
one figures/smc_noise/smoke.json smoke || { echo "FAILED smoke" >> /workspace/status.txt; exit 1; }
say "smoke_rate $(python -c "import json;print(json.load(open('figures/smc_noise/smoke.json'))['traj_per_s_B_batch4096'])")"
# 2) first B run alone: throughput
t0=$(date +%s); one figures/smc_noise/run_B_ep1000_r0.json run B 1000 0 || { echo "FAILED first_B" >> /workspace/status.txt; exit 1; }
t1=$(date +%s); D1=$(( t1 - t0 ))
N1=$(python -c "import json;j=json.load(open('figures/smc_noise/run_B_ep1000_r0.json'));print(j['n_energy_pass1']+j['n_energy_pass2']+j['is_equal_cost']['n'])")
J1=$(python -c "import json;print(json.load(open('figures/smc_noise/run_B_ep1000_r0.json'))['pass1']['J'])")
RATE=$(python -c "print($N1/max($D1,1))"); say "rate_first_B traj=$N1 J=$J1 seconds=$D1 traj_per_s=$RATE"
# stage: name, trajectories estimate, parallel, commands (out|args ...)
stage() { name=$1; traj=$2; par=$3; shift 3
  est=$(python -c "print(int(1.15*$traj/$RATE))"); say "stage $name projected_s=$est left_s=$(left)"
  [ $est -gt $(left) ] && { say "STOP_before $name (projection over cap)"; return 1; }
  pids=(); for j in "$@"; do while [ $(jobs -rp | wc -l) -ge $par ]; do sleep 5; done; ( one ${j%%|*} ${j#*|} ) & pids+=($!); sleep 3; done
  for q in ${pids[@]}; do wait $q; done; say "stage_done $name"; return 0; }
BJ=(); for e in 1000 1250 1500; do for r in 0 1 2; do [ "$e$r" = "10000" ] && continue; BJ+=("figures/smc_noise/run_B_ep${e}_r${r}.json|run B $e $r"); done; done
T_B=$(( 8 * N1 ))
CJ=(); for r in 0 1 2; do CJ+=("figures/smc_noise/run_C_ep250_r${r}.json|run C 250 $r"); done; T_C=$(( 3 * 3 * 2048 * (1 + 5 * 20) ))
AJ=(); for e in 1000 1250 1500; do for r in 0 1 2; do AJ+=("figures/smc_noise/run_A_ep${e}_r${r}.json|run A $e $r"); done; done; T_A=$(( 9 * 3 * 2048 * (1 + 5 * 2) ))
stage SMC_B $T_B 2 "${BJ[@]}" && stage SMC_C $T_C 2 "${CJ[@]}" && stage SMC_A $T_A 2 "${AJ[@]}" && {
  PB=(); for e in 1000 1250 1500; do PB+=("figures/smc_noise/pcn_is_B_ep${e}.json|pcnis B $e"); for rho in 0.0 0.9 0.99 0.999; do PB+=("figures/smc_noise/pcn_B_ep${e}_rho${rho}.json|pcn B $e --rho $rho"); done; done
  stage PCN_B 3850000 3 "${PB[@]}" && {
    PC=("figures/smc_noise/pcn_is_C_ep250.json|pcnis C 250"); for rho in 0.0 0.9 0.99 0.999; do PC+=("figures/smc_noise/pcn_C_ep250_rho${rho}.json|pcn C 250 --rho $rho"); done
    stage PCN_C 1290000 3 "${PC[@]}"; }; }
grep -q "rc=[^0]" $P && echo "SOME_NONZERO_RC" >> /workspace/status.txt
echo DONE >> /workspace/status.txt; say finished
