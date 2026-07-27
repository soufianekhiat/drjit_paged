import drjit as dr
import pytest
import sys


def get_pkg(t):
    with dr.detail.scoped_rtld_deepbind():
        m = pytest.importorskip("paged_ext")
    backend = dr.backend_v(t)
    if backend == dr.JitBackend.LLVM:
        return m.llvm
    elif backend == dr.JitBackend.CUDA:
        return m.cuda
    elif backend == dr.JitBackend.Metal:
        return m.metal


def make_view(pkg, t, page_values, logical_size, page_size):
    pages = [t(v) for v in page_values]
    return pkg.PagedArrayViewF32(pages, logical_size, page_size)


def ref_values(page_values):
    r = []
    for v in page_values:
        r += list(v)
    return r


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test01_construction(t):
    pkg = get_pkg(t)
    UInt32 = sys.modules[t.__module__].UInt32

    # Single page
    view = make_view(pkg, t, [[1, 2, 3, 4]], 4, 4)
    assert view.size() == 4 and view.page_count() == 1

    # Multiple pages with a partial final page
    view = make_view(pkg, t, [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10]], 10, 4)
    assert view.size() == 10 and view.page_count() == 3
    assert view.page_length(0) == 4 and view.page_length(2) == 2
    assert view.power_of_two_page_size()

    # Non-power-of-two page size
    view = make_view(pkg, t, [[1, 2, 3], [4, 5]], 5, 3)
    assert not view.power_of_two_page_size()

    # Empty view
    view = pkg.PagedArrayViewF32()
    assert view.size() == 0 and view.page_count() == 0
    r = pkg.gather(view, UInt32(0, 1), dr.mask_t(t)(False))
    assert dr.all(r == 0)

    # Validation failures
    with pytest.raises(RuntimeError, match='requires'):
        make_view(pkg, t, [[1, 2, 3, 4]], 8, 4)
    with pytest.raises(RuntimeError, match='entries'):
        make_view(pkg, t, [[1, 2, 3, 4], [5, 6, 7]], 8, 4)
    with pytest.raises(RuntimeError, match='entries'):
        make_view(pkg, t, [[1, 2, 3], [5, 6, 7]], 7, 3)
    with pytest.raises(RuntimeError, match='nonzero'):
        make_view(pkg, t, [[1]], 1, 0)
    with pytest.raises(RuntimeError, match='cannot have pages'):
        make_view(pkg, t, [[1]], 0, 4)


@pytest.test_arrays('float32,shape=(*),jit,-diff')
@pytest.mark.parametrize('page_size', [2, 3, 4, 5, 8])
def test02_access(t, page_size):
    pkg = get_pkg(t)
    UInt32 = sys.modules[t.__module__].UInt32

    n = 21
    values = [float(i) * 3 + 1 for i in range(n)]
    page_values = [values[i:i + page_size]
                   for i in range(0, n, page_size)]
    view = make_view(pkg, t, page_values, n, page_size)

    # Identity read: first element, page boundaries, last element
    idx = dr.arange(UInt32, n)
    assert dr.all(pkg.gather(view, idx) == t(values))

    # Reversed and strided access patterns
    idx = UInt32(list(reversed(range(n))))
    assert dr.all(pkg.gather(view, idx) == t(list(reversed(values))))

    idx = UInt32(list(range(0, n, 7)))
    assert dr.all(pkg.gather(view, idx) == t([values[i] for i in range(0, n, 7)]))


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test03_masking(t):
    pkg = get_pkg(t)
    UInt32 = sys.modules[t.__module__].UInt32
    Bool = dr.mask_t(t)

    view = make_view(pkg, t, [[1, 2, 3, 4], [5, 6, 7, 8]], 8, 4)

    # Explicitly masked lanes yield zero
    idx = UInt32(0, 3, 4, 7)
    active = Bool(True, False, True, False)
    assert dr.all(pkg.gather(view, idx, active) == t(1, 0, 5, 0))

    # Out-of-range indices are masked away and yield zero, even when the
    # caller-provided mask is true
    idx = dr.arange(UInt32, 16)
    active = idx < 8
    r = pkg.gather(view, idx, dr.full(Bool, True, 16))
    assert dr.all(r == dr.select(active, t([1, 2, 3, 4, 5, 6, 7, 8] + [0] * 8), 0))


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test04_shared_pages(t):
    pkg = get_pkg(t)
    UInt32 = sys.modules[t.__module__].UInt32

    page = t(9, 8, 7, 6)
    view = pkg.PagedArrayViewF32([page, page], 8, 4)

    idx = dr.arange(UInt32, 8)
    assert dr.all(pkg.gather(view, idx) == t(9, 8, 7, 6, 9, 8, 7, 6))


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test05_update_page(t):
    pkg = get_pkg(t)
    UInt32 = sys.modules[t.__module__].UInt32

    view = make_view(pkg, t, [[1, 2], [3, 4], [5, 6]], 6, 2)
    idx = dr.arange(UInt32, 6)
    assert dr.all(pkg.gather(view, idx) == t(1, 2, 3, 4, 5, 6))

    view.update_page(1, t(30, 40))
    assert dr.all(pkg.gather(view, idx) == t(1, 2, 30, 40, 5, 6))

    with pytest.raises(RuntimeError, match='out of bounds'):
        view.update_page(3, t(1, 2))
    with pytest.raises(RuntimeError, match='entries'):
        view.update_page(0, t(1, 2, 3))


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test06_lifetime(t):
    pkg = get_pkg(t)
    UInt32 = sys.modules[t.__module__].UInt32
    import gc

    pages = [t(1, 2, 3, 4), t(5, 6, 7, 8)]
    view = pkg.PagedArrayViewF32(pages, 8, 4)

    # The view owns page references; dropping all Python handles is safe
    del pages
    gc.collect()

    idx = dr.arange(UInt32, 8)
    assert dr.all(pkg.gather(view, idx) == t(1, 2, 3, 4, 5, 6, 7, 8))


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test07_packet_gather(t):
    pkg = get_pkg(t)
    UInt32 = sys.modules[t.__module__].UInt32

    # Pages of 4 entries; vector i occupies entries [3i, 3i+2], so vector 1
    # straddles the boundary between pages 0 and 1
    view = make_view(pkg, t, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], 12, 4)

    x, y, z = pkg.gather3(view, UInt32(0, 1, 2, 3), dr.mask_t(t)(True))
    assert dr.all(x == t(0, 3, 6, 9))
    assert dr.all(y == t(1, 4, 7, 10))
    assert dr.all(z == t(2, 5, 8, 11))


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test08_symbolic_loop(t):
    pkg = get_pkg(t)
    UInt32 = sys.modules[t.__module__].UInt32

    view = make_view(pkg, t, [[1, 2, 3, 4], [5, 6, 7, 8]], 8, 4)

    # Sum view[0..8) per lane inside a symbolic loop. The page table must be
    # read at run time rather than being unrolled into the loop.
    i = dr.zeros(UInt32, 4)
    acc = dr.zeros(t, 4)

    def cond(i, acc):
        return i < 8

    def body(i, acc):
        acc += pkg.gather(view, i)
        return i + 1, acc

    i, acc = dr.while_loop((i, acc), cond, body)
    assert dr.all(acc == 36)


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test09_symbolic_cond(t):
    pkg = get_pkg(t)
    UInt32 = sys.modules[t.__module__].UInt32

    view = make_view(pkg, t, [[1, 2, 3, 4], [5, 6, 7, 8]], 8, 4)

    idx = UInt32(0, 2, 5, 7)

    def true_fn(idx):
        return pkg.gather(view, idx)

    def false_fn(idx):
        return t(0)

    r = dr.if_stmt((idx,), idx >= 2, true_fn, false_fn)
    assert dr.all(r == t(0, 3, 6, 8))


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test10_kernel_reuse(t):
    pkg = get_pkg(t)
    UInt32 = sys.modules[t.__module__].UInt32

    # Replacing pages (same structure) must not lead to kernel recompilation:
    # page addresses are read from the pointer table at run time and must not
    # be baked into the generated code.
    view = make_view(pkg, t, [[1, 2, 3, 4], [5, 6, 7, 8]], 8, 4)
    idx = dr.arange(UInt32, 8)

    with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
        r1 = pkg.gather(view, idx)
        dr.eval(r1)
        h1 = [h for h in dr.kernel_history() if h['type'] == dr.KernelType.JIT]

        # Replace both pages with fresh allocations and different contents
        view.update_page(0, t(10, 20, 30, 40))
        view.update_page(1, t(50, 60, 70, 80))

        r2 = pkg.gather(view, idx)
        dr.eval(r2)
        h2 = [h for h in dr.kernel_history() if h['type'] == dr.KernelType.JIT]

    assert dr.all(r1 == t(1, 2, 3, 4, 5, 6, 7, 8))
    assert dr.all(r2 == t(10, 20, 30, 40, 50, 60, 70, 80))

    assert len(h1) == 1 and len(h2) == 1
    assert h1[0]['hash'] == h2[0]['hash']
    assert h2[0]['cache_hit']


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test11_page_allocation_identity_independent_kernels(t):
    pkg = get_pkg(t)
    UInt32 = sys.modules[t.__module__].UInt32

    # Two views with identical logical size and page size, but a different
    # number of (shared) backing allocations, must compile to the same
    # kernel: kernel identity is independent of page allocation identity
    # and sharing.
    n, page_size = 32, 4
    values = [float(i) for i in range(n)]
    chunks = [values[i:i + page_size] for i in range(0, n, page_size)]

    # View A: eight distinct pages; view B: one page reused eight times
    view_a = make_view(pkg, t, chunks, n, page_size)
    page = t(chunks[0])
    view_b = pkg.PagedArrayViewF32([page] * 8, n, page_size)

    idx = dr.arange(UInt32, n)

    with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
        ra = pkg.gather(view_a, idx)
        dr.eval(ra)
        ha = [h for h in dr.kernel_history() if h['type'] == dr.KernelType.JIT]

        rb = pkg.gather(view_b, idx)
        dr.eval(rb)
        hb = [h for h in dr.kernel_history() if h['type'] == dr.KernelType.JIT]

    assert dr.all(ra == t(values))
    assert dr.all(rb == t(chunks[0] * 8))
    assert len(ha) == 1 and len(hb) == 1
    assert ha[0]['hash'] == hb[0]['hash']


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test12_single_page(t):
    pkg = get_pkg(t)
    UInt32 = sys.modules[t.__module__].UInt32
    Bool = dr.mask_t(t)

    # A single-entry pointer table takes special code paths, since ordinary
    # gathers from a size-1 source are elided rather than compiled into
    # memory reads
    view = make_view(pkg, t, [[3, 5, 7, 9, 11, 13, 15, 17]], 8, 8)

    # Lane-varying indices
    idx = dr.arange(UInt32, 8)
    assert dr.all(pkg.gather(view, idx) == t(3, 5, 7, 9, 11, 13, 15, 17))

    # Single-element index array
    assert dr.all(pkg.gather(view, UInt32(5)) == t(13))

    # Lane-varying mask
    active = Bool(True, False, True, False, True, False, True, False)
    r = pkg.gather(view, idx, active)
    assert dr.all(r == t(3, 0, 7, 0, 11, 0, 15, 0))

    # Out-of-range lanes yield zero
    idx = dr.arange(UInt32, 12)
    r = pkg.gather(view, idx)
    assert dr.all(r == t([3, 5, 7, 9, 11, 13, 15, 17] + [0] * 4))


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test13_expression_outlives_view(t):
    pkg = get_pkg(t)
    UInt32 = sys.modules[t.__module__].UInt32
    import gc

    # An unevaluated gather expression must keep the referenced pages alive
    # even after the view (which owns the only page references) is gone
    view = make_view(pkg, t, [[1, 2, 3, 4], [5, 6, 7, 8]], 8, 4)
    r = pkg.gather(view, dr.arange(UInt32, 8))

    del view
    gc.collect()

    # Try to provoke reuse of any prematurely freed page storage
    for _ in range(4):
        dr.eval(dr.full(t, 999.0, 4))

    dr.eval(r)
    assert dr.all(r == t(1, 2, 3, 4, 5, 6, 7, 8))


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test14_update_page_before_eval(t):
    pkg = get_pkg(t)
    UInt32 = sys.modules[t.__module__].UInt32

    # Replacing a page must not invalidate previously created (and not yet
    # evaluated) expressions: each pointer table snapshots its pages
    view = make_view(pkg, t, [[1, 2, 3, 4], [5, 6, 7, 8]], 8, 4)
    idx = dr.arange(UInt32, 8)

    r_old = pkg.gather(view, idx)

    view.update_page(0, t(10, 20, 30, 40))
    view.update_page(1, t(50, 60, 70, 80))

    # Try to provoke reuse of any prematurely freed page storage
    for _ in range(4):
        dr.eval(dr.full(t, 999.0, 4))

    r_new = pkg.gather(view, idx)
    dr.eval(r_old, r_new)

    assert dr.all(r_old == t(1, 2, 3, 4, 5, 6, 7, 8))
    assert dr.all(r_new == t(10, 20, 30, 40, 50, 60, 70, 80))


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test15_lifetime_stress(t):
    pkg = get_pkg(t)
    UInt32 = sys.modules[t.__module__].UInt32
    import gc

    # Stress the retainer cleanup: pointer tables reference their pages, and
    # every reference taken while building, replacing, and destroying views
    # and expressions must be released again
    shared_page = t(1, 2, 3, 4)
    dr.eval(shared_page)
    baseline = pkg.jit_ref_count(shared_page)

    idx = dr.arange(UInt32, 8)
    for i in range(1000):
        view = pkg.PagedArrayViewF32([shared_page, t(5, 6, 7, 8)], 8, 4)
        r1 = pkg.gather(view, idx)
        view.update_page(1, t(50, 60, 70, 80))
        r2 = pkg.gather(view, idx)
        del view, r1, r2

    gc.collect()
    assert pkg.jit_ref_count(shared_page) == baseline

