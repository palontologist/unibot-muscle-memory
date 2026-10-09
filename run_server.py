"""Serve the Muscle Memory policy on N consecutive UniBot evaluator ports.

    UNIBOT_SUBMISSION_TOKEN=<token> UNIBOT_CONTROL_SPACE=joint \
        python run_server.py [N]

Template "Approach 1": one process per port, each loading its own copy of the
model. The policy is ~0.8M params and CPU-resident, so N copies are cheap and
per-step latency stays flat.

The number of exposed ports must equal the number of test robots N, and they
must be the N consecutive ports starting at 8765. Pass N as the argument; it
defaults to 1.

    python run_server.py 3      # serve 8765, 8766, 8767
"""

import logging
import multiprocessing as mp
import os
import signal
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # repo root, for the `policy` package

from policy.muscle_memory_policy import MuscleMemorySubmission  # noqa: E402
from policy.web_policy import PolicyService  # noqa: E402

PORT_BASE = 8765


def serve_one(port: int) -> None:
    """Build the policy and serve it on 0.0.0.0:port."""
    policy = MuscleMemorySubmission()
    safe_meta = {k: v for k, v in policy.metadata.items() if k != "token"}
    print(f"Serving MuscleMemorySubmission on ws://0.0.0.0:{port}", flush=True)
    print("  metadata =", safe_meta, flush=True)
    PolicyService(policy, host="0.0.0.0", port=port).run_forever()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [unibot-server] %(message)s")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if n < 1:
        raise SystemExit("usage: python run_server.py [N]  (N = number of test robots)")
    if os.environ.get("UNIBOT_CONTROL_SPACE", "joint") != "joint":
        raise SystemExit("this submission implements the 'joint' control space only")

    print(f"robots = {n}; ports {PORT_BASE}..{PORT_BASE + n - 1}", flush=True)
    if n == 1:
        serve_one(PORT_BASE)
        return

    ctx = mp.get_context("spawn")
    procs = []
    for i in range(n):
        p = ctx.Process(target=serve_one, args=(PORT_BASE + i,), daemon=False)
        p.start()
        procs.append(p)

    def shutdown(*_):
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()