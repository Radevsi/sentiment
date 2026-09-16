#!/usr/bin/env python3
"""Read-only cluster inspection. Does not list files, read credentials, or start jobs."""
import argparse
import datetime
import os
import shutil
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("paths", nargs="*", help="Optional existing project/scratch directories to inspect")
args = parser.parse_args()
started = time.monotonic()
paths = list(dict.fromkeys([os.getcwd(), os.path.expanduser("~"), *args.paths]))
commands = [
    ("Identity and system", ["hostname"]),
    ("Group memberships", ["id"]),
    ("Filesystem capacity (NOT your quota)", ["df", "-h", *paths]),
    ("Filesystem type", ["df", "-T", *paths]),
    ("Filesystem inode capacity (NOT your quota)", ["df", "-i", *paths]),
    ("User quota (may be unsupported)", ["quota", "-s"]),
    ("Group quota (may be unsupported)", ["quota", "-g", "-s"]),
    ("Slurm queues, time limits, CPUs, memory, GPUs", ["sinfo", "-o", "%P %a %l %c %m %G"]),
    ("Slurm account associations (may be restricted)", ["sacctmgr", "-n", "show", "assoc", "user="+os.environ.get("USER", ""), "format=Cluster,Account,Partition,QOS", "-P"]),
    ("PBS queues if present", ["qstat", "-Q"]),
    ("LSF queues if present", ["bqueues"]),
    ("GPU visibility on THIS host only", ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"]),
    ("Python version", ["python3", "--version"]),
    ("tmux version", ["tmux", "-V"]),
]
for number, (label, command) in enumerate(commands, 1):
    step = time.monotonic()
    print(f"\n[{datetime.datetime.now().isoformat(timespec='seconds')}] {number}/{len(commands)} {label}", flush=True)
    if shutil.which(command[0]):
        try:
            result = subprocess.run(command, timeout=20, check=False)
            if result.returncode:
                print(f"Unavailable/failed: exit {result.returncode}")
        except subprocess.TimeoutExpired:
            print("Timed out after 20 seconds; skipped")
    else:
        print(f"{command[0]} not installed or not on PATH; skipped")
    print(f"Step elapsed: {time.monotonic()-step:.1f}s", flush=True)
print("\nStorage hints (only named path variables):")
for key in ("SCRATCH", "WORK", "PROJECT", "TMPDIR", "SLURM_TMPDIR"):
    if key in os.environ:
        print(f"{key}={os.environ[key]}")
print(f"Overall elapsed: {time.monotonic()-started:.1f}s")
print("No quota output does NOT mean unlimited storage. Ask the lab/admin about project quota and scratch purge rules.")
