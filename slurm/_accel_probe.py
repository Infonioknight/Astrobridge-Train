"""Tiny diagnostic: prove accelerate's multi-node rendezvous works before a real run.

Launched through the same slurm/_accel_node.sh path the training stages use, so a pass here
means the launcher, the rank math and the NCCL handshake are all good. Prints one line per
rank and does a real all-reduce — a rendezvous that half-forms HANGS rather than erroring,
so the collective is the part that actually proves it.
"""
import os, torch
from accelerate import Accelerator

acc = Accelerator()
t = torch.tensor([acc.process_index + 1], device=acc.device, dtype=torch.float32)
total = acc.reduce(t.clone(), reduction="sum").item()
expected = acc.num_processes * (acc.num_processes + 1) / 2
print(
    f"[probe] rank={acc.process_index}/{acc.num_processes} "
    f"local={acc.local_process_index} host={os.uname().nodename.split('.')[0]} "
    f"device={acc.device} allreduce={total:.0f} expected={expected:.0f} "
    f"{'OK' if abs(total-expected) < 1e-6 else 'MISMATCH'}",
    flush=True,
)
acc.wait_for_everyone()
if acc.is_main_process:
    print(f"[probe] RENDEZVOUS OK across {acc.num_processes} ranks", flush=True)
