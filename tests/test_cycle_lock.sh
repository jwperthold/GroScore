#!/bin/bash
# Exercise claim_cycle, extracted verbatim from job_fe.run, against every case.
W=/tmp/locktest
rm -rf "$W"; mkdir -p "$W"; cd "$W" || exit 1

eval "$(sed -n '/^  CYCLE_LOCK=\"\"$/,/^  }$/p' /home/jwperthold/GroScore/job_fe.run \
        | sed 's/^  //')"
eval "$(sed -n '/^  claim_cycle() {$/,/^  }$/p' /home/jwperthold/GroScore/job_fe.run \
        | sed 's/^  //')"

fail=0
ck() { printf "  %-56s %s\n" "$1" "$([ "$2" = "$3" ] && echo pass || { echo "FAIL(got $2 want $3)"; fail=1; })"; }

echo "=== 1. a free cycle is claimed ==="
claim_cycle 1 >/dev/null; ck "claim succeeds" "$?" "0"
ck "lock exists" "$([ -d results_fe_c1.lock ] && echo y || echo n)" "y"
ck "owner records host:pid:job" \
   "$(awk -F: '{print NF}' results_fe_c1.lock/owner)" "3"
release_cycle_lock
ck "release removes it" "$([ -d results_fe_c1.lock ] && echo y || echo n)" "n"

echo
echo "=== 2. a LIVE local owner is respected ==="
mkdir -p results_fe_c2.lock; echo "$(hostname):$$:-" > results_fe_c2.lock/owner
claim_cycle 2 >/dev/null; ck "claim refused while pid is alive" "$?" "1"
ck "lock left intact" "$([ -d results_fe_c2.lock ] && echo y || echo n)" "y"
rm -rf results_fe_c2.lock

echo
echo "=== 3. a DEAD local owner is reclaimed ==="
mkdir -p results_fe_c3.lock; echo "$(hostname):999999:-" > results_fe_c3.lock/owner
claim_cycle 3 >/dev/null; ck "claim succeeds after reclaim" "$?" "0"
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
ck "lock left intact" "$([ -d results_fe_c6.lock ] && echo y || echo n)" "y"

echo
[ "$fail" = 0 ] && echo "all lock checks passed" || echo "LOCK CHECKS FAILED"
exit "$fail"
