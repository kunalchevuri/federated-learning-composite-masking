"""
Run a fixed list of full-protocol jobs (colab/jobs.json) on Colab, writing each finished
run straight to Google Drive so a disconnect loses at most the jobs in progress.

Usage (from the unzipped bundle directory):  python colab/colab_run.py /content/drive/MyDrive/masking_runs/
"""
import json
import os
import sys
import time
from multiprocessing import get_context

sys.path.insert(0, 'src')
import revision_experiments as rx  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else '/content/drive/MyDrive/masking_runs/'
os.makedirs(OUT, exist_ok=True)
rx.OUT_DIR = OUT


def run(job):
    rx.OUT_DIR = OUT
    return rx.run_job(tuple(job))


if __name__ == '__main__':
    jobs = json.load(open('colab/jobs.json'))
    n = int(os.environ.get('REV_WORKERS', os.cpu_count()))
    print(f"{len(jobs)} jobs on {n} workers -> {OUT}", flush=True)
    t0 = time.time()
    with get_context('fork').Pool(n) as pool:
        for path, status, secs in pool.imap_unordered(run, jobs):
            print(f"[{time.time() - t0:7.0f}s] {status:6s} {secs:6.0f}s  {os.path.basename(path)}", flush=True)
    print('all done')
