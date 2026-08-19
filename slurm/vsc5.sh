#!/bin/bash
#SBATCH -J GroScore
#SBATCH --partition=zen2_0256_a40x2
#SBATCH --qos=zen2_0256_a40x2
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1

# SLOW NODES, EXCLUDED. Measured, not guessed: every switching leg writes its own
# ns/day, and over test13/14/15 the a40x2 pool splits cleanly in two -- a fast group
# around 980 ns/day and these six around 377, i.e. 2.6x slower for the same work.
# They are not merely unlucky: n3066-007, -010 and -011 came out slow in all three
# runs, on 90-93% of every leg they ever ran. Nodes that are SOMETIMES slow
# (n3066-003, -005, -006, -014, -015, n3067-001, -002) are contended rather than
# bad and are deliberately left in -- excluding them would move the contention
# rather than remove it, and shrink the pool for nothing.
#
# This gives up about 14% of the pool, which those nodes were running at 38% speed.
#
# The list is a measurement and measurements rot. Regenerate it after any cluster
# change rather than trusting this line:
#     python3 utils/fe_node_report.py test13 test14 test15 --exclude-line
#
# n3067-017 is NOT here. It threw one node failure, but at 1013 ns/day over 38 legs
# it is among the fastest in the pool, and one node failure is the cluster's
# background rate, not evidence about a node.
#SBATCH --exclude=n3066-002,n3066-007,n3066-008,n3066-010,n3066-011,n3066-012
