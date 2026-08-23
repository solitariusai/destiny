# Copyright 2026 Shinapri
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Benchmark Taktiny checkpoint save on a real accelerator.

Runs the streamed save path across shard sizes and prefetch modes, parses
the ``[taktiny]`` phase logs, and prints one table row per configuration:

    python benchmarks/bench_checkpoint_save.py \
        --repo google/gemma-3-4b-it --out /kaggle/working

Metrics per run: total wall time, D2H seconds, sync seconds, write seconds,
run count, median run MB, effective D2H/write MB/s, peak host RSS.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import shutil
import tempfile
import threading
import time

SHARD_SIZES = ('512MB', '1GB', '2GB', '4GB')
SHARD_LINE = re.compile(
    r'\[taktiny\] shard \d+/\d+: D2H (?P<d2h>[\d.]+)s '
    r'\(\+sync (?P<sync>[\d.]+)s\) \| (?P<runs>\d+) runs, '
    r'largest (?P<largest>[\d.]+)MB, median (?P<median>[\d.]+)MB'
)
WRITE_LINE = re.compile(r'\[taktiny\] wrote \S+ in (?P<write>[\d.]+)s')
TOTAL_LINE = re.compile(
    r'\[taktiny\] checkpoint written to .*: '
    r'(?P<gb>[\d.]+) GB in (?P<total>[\d.]+)s'
)


def _peak_rss_mb(stop: threading.Event, result: dict) -> None:
    """Sample VmRSS from /proc into ``result['peak']`` until stopped."""
    peak = 0.0
    while not stop.is_set():
        try:
            with open('/proc/self/status') as status:
                for line in status:
                    if line.startswith('VmRSS:'):
                        peak = max(peak, float(line.split()[1]) / 1024)
                        break
        except OSError:
            break
        time.sleep(0.05)
    result['peak'] = peak


def parse_log(text: str) -> dict:
    stats = {
        'd2h': 0.0, 'sync': 0.0, 'write': 0.0, 'total': None,
        'runs': 0, 'median_mb': [], 'shards': 0,
    }
    for match in SHARD_LINE.finditer(text):
        stats['d2h'] += float(match.group('d2h'))
        stats['sync'] += float(match.group('sync'))
        stats['runs'] += int(match.group('runs'))
        stats['median_mb'].append(float(match.group('median')))
        stats['shards'] += 1
    for match in WRITE_LINE.finditer(text):
        stats['write'] += float(match.group('write'))
    total = TOTAL_LINE.search(text)
    if total is not None:
        stats['total'] = float(total.group('total'))
    return stats


def run_one(model, out_dir: str, shard_size: str, prefetch: bool) -> dict:
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.environ['TAKTINY_SAVE_PREFETCH'] = '1' if prefetch else '0'

    stop = threading.Event()
    rss = {'peak': 0.0}
    sampler = threading.Thread(target=_peak_rss_mb, args=(stop, rss))
    sampler.start()

    buffer = io.StringIO()
    start = time.perf_counter()
    try:
        with contextlib.redirect_stdout(buffer):
            model.save_pretrained(out_dir, max_shard_size=shard_size)
        wall = time.perf_counter() - start
    finally:
        stop.set()
        sampler.join()

    stats = parse_log(buffer.getvalue())
    gb = float(re.search(r'([\d.]+) GB', buffer.getvalue()).group(1)) \
        if ' GB' in buffer.getvalue() else 0.0
    row = {
        'size': shard_size,
        'mode': 'prefetch' if prefetch else 'seq',
        'shards': stats['shards'],
        'wall': stats['total'] if stats['total'] else wall,
        'd2h': stats['d2h'],
        'sync': stats['sync'],
        'write': stats['write'],
        'runs': stats['runs'],
        'median_mb': (
            sorted(stats['median_mb'])[len(stats['median_mb']) // 2]
            if stats['median_mb'] else 0.0
        ),
        'd2h_mbs': stats['d2h'] * 1000 / max(gb, 1e-9),
        'write_mbs': stats['write'] * 1000 / max(gb, 1e-9),
        'peak_rss_mb': rss['peak'],
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='google/gemma-3-4b-it')
    parser.add_argument('--out', default=None)
    parser.add_argument(
        '--sizes',
        default=','.join(SHARD_SIZES),
        help='comma-separated shard sizes to sweep',
    )
    parser.add_argument(
        '--modes',
        default='seq,prefetch',
        help='comma-separated: seq,prefetch',
    )
    parser.add_argument(
        '--dtype', default='bfloat16', help='model dtype for from_pretrained'
    )
    args = parser.parse_args()

    import jax

    print(f'JAX devices: {jax.devices()}', flush=True)

    from taktiny.maestro import Maestro

    print(f'Loading {args.repo}...', flush=True)
    model = Maestro.from_pretrained(args.repo, dtype=args.dtype)
    print(model.placement_report(), flush=True)

    out_root = args.out or tempfile.mkdtemp(prefix='taktiny-bench-')
    sizes = [s.strip() for s in args.sizes.split(',') if s.strip()]
    modes = [m.strip() for m in args.modes.split(',') if m.strip()]

    header = (
        f"{'mode':<9}{'size':<7}{'shards':>6}{'wall_s':>8}{'d2h_s':>7}"
        f"{'sync_s':>7}{'write_s':>8}{'runs':>5}{'medMB':>7}"
        f"{'d2h_MB/s':>10}{'wr_MB/s':>9}{'rss_MB':>8}"
    )
    rows = []
    for mode in modes:
        for size in sizes:
            out_dir = os.path.join(out_root, f'bench-{mode}-{size}')
            row = run_one(
                model, out_dir, size, prefetch=(mode == 'prefetch')
            )
            rows.append(row)
            print(header)
            print(
                f"{row['mode']:<9}{row['size']:<7}{row['shards']:>6}"
                f"{row['wall']:>8.1f}{row['d2h']:>7.1f}{row['sync']:>7.1f}"
                f"{row['write']:>8.1f}{row['runs']:>5}{row['median_mb']:>7.0f}"
                f"{row['d2h_mbs']:>10.0f}{row['write_mbs']:>9.0f}"
                f"{row['peak_rss_mb']:>8.0f}",
                flush=True,
            )
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)

    print('\nDone. Rows are also usable for the shard-size sweep report.')


if __name__ == '__main__':
    main()
