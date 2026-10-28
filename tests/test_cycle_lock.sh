#!/bin/bash
# The locks of job_fe.run, extracted verbatim and run against every case.
#
#   1-6  claim_cycle: one process per cycle. A live owner is respected, a dead one
#        on this host is reclaimed, one on another host only when squeue says its
#        SLURM job is gone, and never without a job id.
#   7-11 the setup lock (job_run.lock): one setup or legacy job per structure
#        directory, taken by --setup and by legacy mode, never by --cycle, and
#        released on exit. It replaced a file-time guess that counted the files the
#        submitter writes, so a fresh --run-local --sequential start exited 0 at
#        once and the local runner logged it 'done' (audit of 2026-10-04).
#   12   a TERM (walltime, scancel, local-runner shutdown) stops the job with 143
#        and releases its locks, in the setup and in the cycle phase. A trap that
#        released without exiting let bash run the rest of the script and exit 0.
#   13   squeue must answer (review S1): a controller error keeps the lock of an
#        owner on another host, and of a live pid on this one; 'Invalid job id' and
#        a finished state count as gone, a running job does not.
#   14   reclaiming is atomic (review S2): two processes reclaiming one stale lock
#        never both get it, also when the second judged the owner dead before the
#        first took the lock.
#   15   the running leg is part of the lock (review S3): a dead owner whose leg
#        still runs keeps its lock; a TERM stops the leg at once (run_bg, stop_leg).
#   16   the archive counts cycles 1..NUMCYCLES only (review S5).
#   17   three processes and one stale lock (review V3): reclaimers take turns under
#        LOCK.reclaim, so a late reclaimer leaves the lock another one has retaken
#        and a third start does not get it too; a fresh mutex keeps other
#        reclaimers out, one older than 60 s is removed; a lock that cannot be put
#        back is reported, and nothing is left moved aside.
#   18   no cycle starts while the structure is being archived (.archive.lock).
#   Both runners carry the same lock functions; 13-15 and 17 run job.run's as well.
#   The traps of job_fe.run also print the task's end line and stop an archive
#   (task_banner, archive_abort); those functions are loaded too, with a stub
#   nvidia-smi, so the traps run as written.
#
# The checks used to set their failure flag inside a command substitution, i.e. in
# a subshell, so this script reported 'all lock checks passed' whatever happened.
# Each check now counts in the shell that exits.
#
# Standalone: bash tests/test_cycle_lock.sh   (exit 0 = all passed)
JOB="$(cd "$(dirname "$0")/.." && pwd)/job_fe.run"
CJOB="$(cd "$(dirname "$0")/.." && pwd)/job.run"
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
cd "$W" || exit 1

# fn FILE NAME: the top-level bash function NAME of FILE, verbatim.
fn() { sed -n "/^$2() {\$/,/^}\$/p" "$1"; }
LOCK_FNS="proc_start lock_put leg_alive take_lock run_bg stop_leg"
TRAP_FNS="gpu_report task_banner archive_note archive_abort"

# The cycle lock and its release, the shared take_lock and its helpers, and the
# setup-lock block. The functions the traps call besides the locks come too, and a
# stub stands in for nvidia-smi.
mkdir -p "$W/nvbin"
printf '#!/bin/sh\necho "0, stub GPU, 00000000:01:00.0, 1 MiB, 2 MiB, 0 %%"\n' > "$W/nvbin/nvidia-smi"
chmod +x "$W/nvbin/nvidia-smi"
export PATH="$W/nvbin:$PATH"
eval "$(for f in $LOCK_FNS $TRAP_FNS; do fn "$JOB" $f; done)"
eval "$(sed -n '/^  CYCLE_LOCK=\"\"$/,/^  }$/p' "$JOB" | sed 's/^  //')"
eval "$(sed -n '/^  claim_cycle() {$/,/^  }$/p' "$JOB" | sed 's/^  //')"
# Only the first such block, and only if it is the lock: run verbatim, anything else
# under that condition would start a setup.
JOBLOCK_BLOCK=$(sed -n '/^JOB_LOCK=""$/,/^}$/p' "$JOB"; \
                sed -n '/^if \[\[ "\$MODE" != "cycle" \]\]; then$/,/^fi$/{p;/^fi$/q}' "$JOB")
case "$JOBLOCK_BLOCK" in
  *release_job_lock*take_lock*job_run.lock*) ;;
  *) JOBLOCK_BLOCK="" ;;
esac

fail=0
ck() {
  if [ "$2" = "$3" ]; then
    printf "  %-60s pass\n" "$1"
  else
    printf "  %-60s FAIL(got %s want %s)\n" "$1" "$2" "$3"
    fail=1
  fi
}
has() { [ -d "$1" ] && echo y || echo n; }

echo "=== 1. a free cycle is claimed ==="
claim_cycle 1 >/dev/null; ck "claim succeeds" "$?" "0"
ck "lock exists" "$(has results_fe_c1.lock)" "y"
ck "owner records host:pid:job" \
   "$(awk -F: '{print NF}' results_fe_c1.lock/owner 2>/dev/null)" "3"
release_cycle_lock
ck "release removes it" "$(has results_fe_c1.lock)" "n"

echo
echo "=== 2. a LIVE local owner is respected ==="
mkdir -p results_fe_c2.lock; echo "$(hostname):$$:-" > results_fe_c2.lock/owner
claim_cycle 2 >/dev/null; ck "claim refused while pid is alive" "$?" "1"
ck "lock left intact" "$(has results_fe_c2.lock)" "y"
rm -rf results_fe_c2.lock

echo
echo "=== 3. a DEAD local owner is reclaimed ==="
mkdir -p results_fe_c3.lock; echo "$(hostname):4999999:-" > results_fe_c3.lock/owner
claim_cycle 3 >/dev/null; ck "claim succeeds after reclaim" "$?" "0"
ck "  and the owner is now this shell" "$(cut -d: -f2 results_fe_c3.lock/owner 2>/dev/null)" "$$"
release_cycle_lock

echo
echo "=== 4. THE GAP: another host, hard-killed, no SLURM info ==="
mkdir -p results_fe_c4.lock; echo "othernode:12345:-" > results_fe_c4.lock/owner
claim_cycle 4 >/dev/null; ck "still refused (cannot prove it is dead)" "$?" "1"
rm -rf results_fe_c4.lock

echo
echo "=== 5. another host, but a SLURM job id that is NOT queued ==="
mkdir -p bin
cat > bin/squeue <<'EOF'
#!/bin/bash
exit 0          # in the queue: nothing printed, so the job is gone
EOF
chmod +x bin/squeue
OLDPATH=$PATH
PATH="$W/bin:$PATH"
mkdir -p results_fe_c5.lock; echo "othernode:12345:987654" > results_fe_c5.lock/owner
claim_cycle 5 >/dev/null; ck "reclaimed via squeue" "$?" "0"
release_cycle_lock

echo
echo "=== 6. another host, SLURM job STILL queued ==="
cat > bin/squeue <<'EOF'
#!/bin/bash
echo "987654 gpu fe_2KTF jwp R 1:23 1 n3066-001"
EOF
chmod +x bin/squeue
mkdir -p results_fe_c6.lock; echo "othernode:12345:987654" > results_fe_c6.lock/owner
claim_cycle 6 >/dev/null; ck "refused while the job is running" "$?" "1"
ck "lock left intact" "$(has results_fe_c6.lock)" "y"
PATH=$OLDPATH

# The setup-lock block runs in a subshell per case, as the top of job_fe.run would:
# it may exit, and its EXIT trap must release the lock when the subshell ends.
setup_block() {   # setup_block MODE DIR -> prints REACHED if the job went on
  [ -n "$JOBLOCK_BLOCK" ] || { echo "NO LOCK BLOCK"; return 2; }
  ( cd "$2" && MODE=$1 && eval "$JOBLOCK_BLOCK" && echo "REACHED $(cat job_run.lock/owner 2>/dev/null)" )
}

echo
echo "=== 7. --setup in a free directory takes job_run.lock and releases it ==="
ck "the setup lock block is extractable" "$([ -n "$JOBLOCK_BLOCK" ] && echo y || echo n)" "y"
mkdir -p s7
out=$(setup_block setup s7); rc=$?
ck "the job goes on" "$rc $(echo "$out" | grep -c REACHED)" "0 1"
ck "  holding job_run.lock as host:pid:job" "$(echo "$out" | awk '/REACHED/{print split($2, a, ":")}')" "3"
ck "  and the lock is gone once the job exits" "$(has s7/job_run.lock)" "n"

echo
echo "=== 8. legacy mode with a LIVE owner exits 1 and touches nothing ==="
mkdir -p s8/job_run.lock; echo "$(hostname):$$:-" > s8/job_run.lock/owner
out=$(setup_block all s8); rc=$?
ck "exit 1, not 0 (the runner must not log it as done)" "$rc" "1"
ck "  the job did not go on" "$(echo "$out" | grep -c REACHED)" "0"
ck "  and says which lock holds it" "$(echo "$out" | grep -c 'holds job_run.lock')" "1"
ck "  the owner's lock is left as it was" "$(cat s8/job_run.lock/owner)" "$(hostname):$$:-"
out=$(setup_block setup s8); rc=$?
ck "--setup is refused the same way" "$rc $(echo "$out" | grep -c REACHED)" "1 0"

echo
echo "=== 9. a DEAD owner of job_run.lock is reclaimed ==="
mkdir -p s9/job_run.lock; echo "$(hostname):4999999:-" > s9/job_run.lock/owner
out=$(setup_block all s9); rc=$?
ck "the job goes on" "$rc $(echo "$out" | grep -c REACHED)" "0 1"
ck "  after saying it reclaimed the lock" "$(echo "$out" | grep -c 'reclaiming stale lock')" "1"
ck "  and releases it at the end" "$(has s9/job_run.lock)" "n"

echo
echo "=== 10. --cycle never takes job_run.lock ==="
mkdir -p s10/job_run.lock; echo "$(hostname):$$:-" > s10/job_run.lock/owner
out=$(setup_block cycle s10); rc=$?
ck "a cycle task goes on while a setup holds the lock" "$rc $(echo "$out" | grep -c REACHED)" "0 1"
ck "  and leaves that lock alone" "$(cat s10/job_run.lock/owner)" "$(hostname):$$:-"

echo
echo "=== 11. the archive leaves job_run.lock out ==="
# The archive step (archive_structure), run on a finished structure directory that
# holds the lock.
eval "$(fn "$JOB" archive_structure)"
mkdir -p arc/S1/job_run.lock; echo "x" > arc/S1/job_run.lock/owner; echo "x" > arc/S1/keep.gs
: > arc/S1/results_fe_c1.done
( cd arc/S1 && STRUCT_ID=S1 && NUMCYCLES=1 && archive_structure ) > /dev/null
ck "the archive is written" "$([ -s arc/S1.tar.gz ] && echo y || echo n)" "y"
ck "  with the structure's files" "$(tar -tzf arc/S1.tar.gz 2>/dev/null | grep -c 'S1/keep.gs')" "1"
ck "  and without job_run.lock" "$(tar -tzf arc/S1.tar.gz 2>/dev/null | grep -c job_run.lock)" "0"

echo
echo "=== 12. a TERM stops the job and releases its locks ==="
# A trap on TERM that does not exit makes bash survive the signal and run the rest of
# the script, then exit 0: SLURM's walltime TERM and the local runner's shutdown TERM
# would let the job go on writing statuses for work it never did.
CYCLE_TRAPS=$(grep "^  trap '.*release_cycle_lock" "$JOB" | sed 's/^  //')
ck "the cycle phase sets its traps (EXIT, INT, TERM)" "$(echo "$CYCLE_TRAPS" | grep -c .)" "3"
term_case() {   # term_case DIR SCRIPT -> "exit-code stopped|after locks-left"; output in DIR/out
  bash -c "cd '$1' && $2" > "$1/out" 2>&1 &
  local pid=$!
  sleep 1; kill -TERM $pid
  wait $pid; local rc=$?
  echo "$rc $([ -f "$1/after" ] && echo after || echo stopped) $(ls -d "$1"/*.lock 2>/dev/null | wc -l)"
}
# The setup job: the lock block verbatim, then a 3 s command it is waiting on.
mkdir -p s12
if [ -n "$JOBLOCK_BLOCK" ]; then
  SCRIPT12="$(declare -f $LOCK_FNS $TRAP_FNS)
MODE=setup
$JOBLOCK_BLOCK
sleep 3; echo x > after"
  got=$(term_case s12 "$SCRIPT12")
else
  got="no lock block"
fi
ck "a setup job hit by TERM exits 143, runs nothing after, holds no lock" "$got" "143 stopped 0"
# A cycle: both locks held, the cycle phase's traps verbatim.
mkdir -p s13
SCRIPT13="$(declare -f $LOCK_FNS $TRAP_FNS)
$(sed -n '/^JOB_LOCK=""$/,/^}$/p' "$JOB")
$(sed -n '/^  CYCLE_LOCK=\"\"$/,/^  }$/p' "$JOB" | sed 's/^  //')
mkdir results_fe_c1.lock job_run.lock; CYCLE_LOCK=results_fe_c1.lock; JOB_LOCK=\$PWD/job_run.lock
$CYCLE_TRAPS
sleep 3; echo x > after"
ck "a cycle hit by TERM exits 143, runs nothing after, holds no lock" "$(term_case s13 "$SCRIPT13")" "143 stopped 0"
ck "  both print the task's end line with exit 143" \
   "$(cat s12/out s13/out 2>/dev/null | grep -c '^Task end: .*, exit 143, after [0-9]* s$')" "2"

# use FILE: the lock functions of that runner, in this shell (so ck counts here).
use() { eval "$(for f in $LOCK_FNS; do fn "$1" $f; done)"; }
sleep 60 & LIVE=$!

echo
echo "=== 13. squeue must answer (review S1) ==="
mkdir -p bin13
cat > bin13/squeue <<'EOF'
#!/bin/bash
case "$SQ_MODE" in
  timeout)   echo "slurm_load_jobs error: Socket timed out on send/recv operation" >&2; exit 1 ;;
  invalid)   echo "slurm_load_jobs error: Invalid job id specified" >&2; exit 1 ;;
  completed) echo "COMPLETED" ;;
  running)   echo "RUNNING" ;;
esac
EOF
chmod +x bin13/squeue
t13() {   # t13 OWNER SQ_MODE -> "rc pid-field-of-the-owner-afterwards"
  rm -rf L13; mkdir L13; echo "$1" > L13/owner
  SQ_MODE=$2 PATH="$W/bin13:$PATH" take_lock "$W/L13" > out13 2>&1
  echo "$? $(cut -d: -f2 L13/owner 2>/dev/null)"
}
for R in "$JOB" "$CJOB"; do
  use "$R"; tag=$(basename "$R")
  ck "$tag: other host, squeue times out: kept" "$(t13 "othernode:12345:4242" timeout)" "1 12345"
  ck "  saying squeue could not answer" "$(grep -c 'squeue could not say' out13)" "1"
  ck "$tag: this host, LIVE pid, squeue times out: kept" "$(t13 "$(hostname):$LIVE:4242" timeout)" "1 $LIVE"
  ck "$tag: 'Invalid job id specified': reclaimed" "$(t13 "othernode:12345:4242" invalid)" "0 $$"
  ck "$tag: squeue lists it COMPLETED: reclaimed" "$(t13 "othernode:12345:4242" completed)" "0 $$"
  ck "$tag: squeue lists it RUNNING: kept" "$(t13 "othernode:12345:4242" running)" "1 12345"
done
use "$JOB"

echo
echo "=== 14. two reclaimers of one stale lock never both hold it (review S2) ==="
for R in "$JOB" "$CJOB"; do
  use "$R"; tag=$(basename "$R")
  both=0
  for t in $(seq 1 150); do
    rm -rf L14 R1 R2; mkdir L14; echo "$(hostname):4999999:-" > L14/owner
    ( take_lock "$W/L14" >/dev/null && echo y > R1 ) & p1=$!
    ( take_lock "$W/L14" >/dev/null && echo y > R2 ) & p2=$!
    wait $p1 $p2
    [ -f R1 ] && [ -f R2 ] && both=$((both+1))
  done
  ck "$tag: 150 trials of two concurrent reclaimers: both held it in" "$both" "0"
  # The second reclaimer judged the owner dead before the first took the lock: its
  # kill -0 answers half a second late (the interleaving that gave both the lock).
  rm -rf L14 RA RB; mkdir L14; echo "$(hostname):4999999:-" > L14/owner
  ( take_lock "$W/L14" >/dev/null; echo "A $?" > RA ) & pa=$!
  ( kill() { sleep 0.5; builtin kill "$@"; }; take_lock "$W/L14" >/dev/null; echo "B $?" > RB ) & pb=$!
  wait $pa $pb
  ck "$tag: the late reclaimer gets nothing, the first keeps it" "$(cat RA RB 2>/dev/null | tr '\n' ' ')" "A 0 B 1 "
  ck "  and the lock holds the first one's owner line" "$(cat L14/owner 2>/dev/null)" "$(hostname):$$:-"
done
use "$JOB"

echo
echo "=== 15. the running leg is part of the lock (review S3) ==="
for R in "$JOB" "$CJOB"; do
  use "$R"; tag=$(basename "$R")
  rm -rf L15; mkdir L15; echo "$(hostname):4999999:-" > L15/owner
  sleep 60 & leg=$!
  echo "$leg $(proc_start $leg)" > L15/leg
  take_lock "$W/L15" > out15 2>&1
  ck "$tag: dead owner, its leg still running: kept" "$? $(cut -d: -f2 L15/owner)" "1 4999999"
  kill $leg; wait $leg 2>/dev/null
  take_lock "$W/L15" > out15 2>&1
  ck "  reclaimed once the leg has ended" "$? $(cut -d: -f2 L15/owner)" "0 $$"
  rm -rf L15; mkdir L15; echo "$(hostname):4999999:-" > L15/owner; echo "$LIVE 1" > L15/leg
  take_lock "$W/L15" > out15 2>&1
  ck "  a leg pid that now names another process does not keep it" "$? $(cut -d: -f2 L15/owner)" "0 $$"
  rm -rf L15
  # A TERM stops a leg started by run_bg at once and waits for it; before, bash ran
  # the trap only once the leg had returned.
  mkdir -p s15$tag
  bash -c "cd s15$tag && $(declare -f $LOCK_FNS $TRAP_FNS)
JOB_LOCK=\$PWD/L; mkdir L
trap 'stop_leg; exit 143' TERM
run_bg sleep 30
echo x > after" &
  p=$!
  sleep 1
  leg=$(cut -d' ' -f1 s15$tag/L/leg 2>/dev/null)
  t0=$(date +%s%N); kill -TERM $p; wait $p; rc=$?; t1=$(date +%s%N)
  ck "$tag: TERM during a run_bg leg: 143 within 3 s, nothing after it" \
     "$rc $(( (t1 - t0) / 1000000000 < 3 ? 1 : 0 )) $([ -f s15$tag/after ] && echo after || echo stopped)" "143 1 stopped"
  ck "  the leg was recorded in the lock, and is gone" \
     "$([ -n "$leg" ] && echo y || echo n) $( [ -n "$leg" ] && kill -0 $leg 2>/dev/null && echo alive || echo gone)" "y gone"
done
use "$JOB"
kill $LIVE 2>/dev/null; wait $LIVE 2>/dev/null

echo
echo "=== 16. the FE archive counts cycles 1..NUMCYCLES (review S5) ==="
mkdir -p ar16/S1
( cd ar16/S1 && for c in $(seq 1 11) $(seq 21 40); do : > results_fe_c$c.done; done
  echo partial > boundfwd1_12.xtc; echo "dG_release_kJ_mol 30.0" > boresch_analytical.gs )
( cd ar16/S1 && NUMCYCLES=20 && STRUCT_ID=S1 && archive_structure ) > /dev/null 2>&1
ck "cycles 1-11 and 21-40 done, -n 20: not archived, cycle 12 left alone" \
   "$([ -f ar16/S1/boundfwd1_12.xtc ] && [ ! -e ar16/S1.tar.gz ] && echo kept || echo archived)" "kept"
( cd ar16/S1 && for c in $(seq 12 20); do : > results_fe_c$c.done; done )
( cd ar16/S1 && NUMCYCLES=20 && STRUCT_ID=S1 && archive_structure ) > /dev/null 2>&1
ck "  and archived once 1-20 are all done" "$([ ! -d ar16/S1 ] && [ -s ar16/S1.tar.gz ] && echo archived || echo kept)" "archived"

echo
echo "=== 17. one stale lock, three processes: one holder, nothing moved aside (review V3) ==="
# The interleaving of v15_work Z/verify/v30_race.sh: reclaimer A; reclaimer B, whose
# kill -0 answers 0.5 s late and whose read of a moved-aside owner takes 1 s; a third
# start C at 0.9 s. Without the mutex B moved aside the lock A had just retaken, C took
# the empty path, B's put-back failed in silence, and A and C both held it, A's lock
# left at L17.stale.<pid>. The owners carry the job ids A, B and C, and a stub squeue
# reports those jobs RUNNING, as for live jobs (a real one calls them invalid ids).
mkdir -p bin17
cat > bin17/squeue <<'EOF'
#!/bin/bash
echo RUNNING
EOF
chmod +x bin17/squeue
for R in "$JOB" "$CJOB"; do
  use "$R"; tag=$(basename "$R")
  rm -rf L17 L17.* r17; mkdir L17; echo "$(hostname):4999999:-" > L17/owner
  ( PATH="$W/bin17:$PATH"; SLURM_JOB_ID=A; take_lock "$W/L17" >/dev/null 2>&1; echo "A $?" >> r17 ) & pa=$!
  ( PATH="$W/bin17:$PATH"; SLURM_JOB_ID=B
    kill() { sleep 0.5; builtin kill "$@"; }
    cat() { [[ "$1" == *".stale."* ]] && sleep 1; command cat "$@"; }
    take_lock "$W/L17" >/dev/null 2>&1; echo "B $?" >> r17 ) & pb=$!
  ( PATH="$W/bin17:$PATH"; SLURM_JOB_ID=C; sleep 0.9; take_lock "$W/L17" >/dev/null 2>&1; echo "C $?" >> r17 ) & pc=$!
  wait $pa $pb $pc
  ck "$tag: A, a late B and C: only A holds it" "$(sort r17 | tr '\n' ' ')" "A 0 B 1 C 1 "
  ck "  the lock names A" "$(cat L17/owner 2>/dev/null)" "$(hostname):$$:A"
  ck "  and nothing is left beside it" "$(ls -d L17.* 2>/dev/null | wc -l)" "0"
  # A fresh mutex: another reclaimer is at work, so this one leaves the lock alone.
  rm -rf L17 L17.*; mkdir L17 L17.reclaim; echo "$(hostname):4999999:-" > L17/owner
  take_lock "$W/L17" > out17 2>&1
  ck "$tag: a fresh L17.reclaim: the lock and the mutex are left alone" \
     "$? $(cut -d: -f2 L17/owner) $(has L17.reclaim)" "1 4999999 y"
  # A mutex left by a reclaimer that died: kept at 50 s, removed at 70 s (limit 60 s).
  touch -d '-50 seconds' L17.reclaim
  take_lock "$W/L17" > out17 2>&1
  ck "  a mutex 50 s old: still left alone" "$? $(cut -d: -f2 L17/owner) $(has L17.reclaim)" "1 4999999 y"
  touch -d '-70 seconds' L17.reclaim
  take_lock "$W/L17" > out17 2>&1
  ck "  70 s old: removed, and the lock reclaimed" "$? $(cut -d: -f2 L17/owner) $(has L17.reclaim)" "0 $$ n"
  # A put-back that fails. B holds no real mutex (as after a stale one was removed
  # twice), reads the dead owner, and its move waits 0.5 s; A reclaims meanwhile; B
  # then moves A's new lock aside, and its put-back waits 0.5 s; C, at 0.7 s, takes
  # the empty path.
  rm -rf L17 L17.* r17; mkdir L17; echo "$(hostname):4999999:-" > L17/owner
  ( PATH="$W/bin17:$PATH"; SLURM_JOB_ID=B
    mkdir() { [[ "$1" == *.reclaim ]] && return 0; command mkdir "$@"; }
    mv() { case "$2|$3" in *"/L17|"*".stale."*|*".stale."*"|"*"/L17") sleep 0.5 ;; esac; command mv "$@"; }
    take_lock "$W/L17" > out17b 2>&1; echo "B $?" >> r17 ) & pb=$!
  ( PATH="$W/bin17:$PATH"; SLURM_JOB_ID=A; sleep 0.2; take_lock "$W/L17" >/dev/null 2>&1; echo "A $?" >> r17 ) & pa=$!
  ( PATH="$W/bin17:$PATH"; SLURM_JOB_ID=C; sleep 0.7; take_lock "$W/L17" >/dev/null 2>&1; echo "C $?" >> r17 ) & pc=$!
  wait $pa $pb $pc
  ck "$tag: a put-back that fails: B says so" "$(grep -c 'could not be put back' out17b)" "1"
  ck "  returns 1, and leaves nothing moved aside" "$(grep '^B' r17) $(ls -d L17.* 2>/dev/null | wc -l)" "B 1 0"
done
use "$JOB"

echo
echo "=== 18. a cycle does not start while the structure is being archived ==="
mkdir -p s18/.archive.lock
out=$(cd s18 && STRUCT_ID=S18 && claim_cycle 3; echo "rc=$?")
ck "claim refused while .archive.lock exists" \
   "$(echo "$out" | grep -c 'S18 is being archived') $(echo "$out" | grep -o 'rc=[0-9]*')" "1 rc=1"
ck "  and the cycle lock it took is released" "$(has s18/results_fe_c3.lock)" "n"
rmdir s18/.archive.lock
out=$(cd s18 && STRUCT_ID=S18 && claim_cycle 3; echo "rc=$?")
ck "claimed once the marker is gone" "$(echo "$out" | grep -o 'rc=[0-9]*') $(has s18/results_fe_c3.lock)" "rc=0 y"

echo
[ "$fail" = 0 ] && echo "all lock checks passed" || echo "LOCK CHECKS FAILED"
exit "$fail"
