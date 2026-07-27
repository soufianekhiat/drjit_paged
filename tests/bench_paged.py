"""
Benchmark comparing ordinary contiguous gathers, paged primal gathers, and
differentiable paged gathers (see drjit/paged.h and tests/paged_ext.cpp).

Usage:

    python bench_paged.py [--backend llvm|cuda] [--size N] [--reps R]

Measures, per page size and access pattern:

- trace time of the first evaluation (includes kernel compilation)
- kernel execution time (after warm-up, using compiled kernels)
- backward execution time for the differentiable variant
- kernel reuse when replacing page pointers

Run from a build directory containing the compiled drjit package and the
paged_ext test extension.
"""

import argparse
import statistics
import sys
import time

import drjit as dr


def timeit(fn, reps):
    times = []
    for _ in range(reps):
        dr.sync_thread()
        start = time.perf_counter()
        fn()
        dr.sync_thread()
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1e3  # ms


def make_indices(UInt32, pattern, n, page_size):
    import random
    random.seed(3)
    if pattern == 'sequential':
        return dr.arange(UInt32, n)
    elif pattern == 'random':
        idx = list(range(n))
        random.shuffle(idx)
        return UInt32(idx)
    elif pattern == 'coherent':
        # Every lane stays within the first page
        return UInt32([i % page_size for i in range(n)])
    elif pattern == 'divergent':
        # Adjacent lanes land on different pages
        return UInt32([(i * page_size + (i % page_size)) % n
                       for i in range(n)])
    else:
        raise ValueError(pattern)


def run(backend, n, reps):
    if backend == 'llvm':
        import drjit.llvm as mod
        pkg_name = 'llvm'
    else:
        import drjit.cuda as mod
        pkg_name = 'cuda'

    import paged_ext
    pkg = getattr(paged_ext, pkg_name)

    Float, UInt32 = mod.Float, mod.UInt32
    values_list = [i * 0.5 + 1.0 for i in range(n)]
    values = Float(values_list)
    dr.eval(values)

    print(f'backend={backend} n={n} reps={reps}')
    print(f'{"page_size":>10} {"pattern":>10} {"variant":>12} '
          f'{"trace_ms":>9} {"exec_ms":>8} {"bwd_ms":>8} {"reuse":>6}')

    for page_size in (256, 1024, 4096, 16384, 65536):
        if page_size * 2 > n:
            continue

        chunks = [Float(values_list[i:i + page_size])
                  for i in range(0, n, page_size)]
        view = pkg.PagedArrayViewF32(chunks, n, page_size)
        dview = pkg.DiffPagedArrayViewF32(chunks, n, page_size)
        dview.enable_grad()

        for pattern in ('sequential', 'random', 'coherent', 'divergent'):
            idx = make_indices(UInt32, pattern, n, page_size)
            dr.eval(idx)

            variants = {
                'contiguous': lambda: dr.eval(dr.gather(Float, values, idx) * 2),
                'paged': lambda: dr.eval(pkg.gather(view, idx) * 2),
                'paged_diff': lambda: dr.eval(pkg.gather_diff(dview, idx) * 2),
            }

            for name, fn in variants.items():
                trace_start = time.perf_counter()
                fn()  # warm-up (includes compilation)
                dr.sync_thread()
                trace_ms = (time.perf_counter() - trace_start) * 1e3
                exec_ms = timeit(fn, reps)

                bwd_ms = float('nan')
                if name == 'paged_diff':
                    def bwd():
                        dview.clear_grad()
                        y = pkg.gather_diff(dview, idx)
                        dr.backward(dr.sum(y))
                        dr.eval(dview.grad())
                    bwd()
                    bwd_ms = timeit(bwd, reps)

                # Kernel reuse when replacing a page pointer
                reuse = '-'
                if name == 'paged':
                    with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
                        fn()
                        dr.kernel_history()
                        view.update_page(0, Float(values_list[0:page_size]))
                        fn()
                        h = [k for k in dr.kernel_history()
                             if k['type'] == dr.KernelType.JIT]
                    reuse = 'yes' if h and all(k['cache_hit'] for k in h) else 'NO'

                print(f'{page_size:>10} {pattern:>10} {name:>12} '
                      f'{trace_ms:>9.3f} {exec_ms:>8.3f} {bwd_ms:>8.3f} {reuse:>6}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', default='llvm', choices=['llvm', 'cuda'])
    parser.add_argument('--size', type=int, default=1 << 20)
    parser.add_argument('--reps', type=int, default=10)
    args = parser.parse_args()

    if args.backend == 'cuda' and not dr.has_backend(dr.JitBackend.CUDA):
        print('CUDA backend unavailable', file=sys.stderr)
        sys.exit(1)

    run(args.backend, args.size, args.reps)
