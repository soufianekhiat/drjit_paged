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


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test16_freeze_replay(t):
    pkg = get_pkg(t)
    UInt32 = sys.modules[t.__module__].UInt32

    holder = pkg.PagedHolderF32([t(1, 2, 3, 4), t(5, 6, 7, 8)], 8, 4)
    idx = dr.arange(UInt32, 8)

    @dr.freeze
    def func(holder, idx):
        return pkg.gather_holder(holder, idx)

    # Record
    r1 = func(holder, idx)
    assert dr.all(r1 == t(1, 2, 3, 4, 5, 6, 7, 8))
    assert func.n_recordings == 1


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test17_diff_jvp_basic(t):
    # Spec 19.1: JVP crossing both page boundaries
    pkg = get_pkg(t)
    m = sys.modules[t.__module__]
    UInt32 = m.UInt32

    dv = pkg.DiffPagedArrayViewF32([t(1, 2, 3), t(4, 5, 6), t(7, 8)], 8, 3)
    dv.enable_grad()
    dv.set_grad(t(10, 20, 30, 40, 50, 60, 70, 80))

    y = pkg.gather_diff(dv, UInt32(2, 3, 5, 6))
    assert dr.all(dr.detach(y) == m.ad.Float(3, 4, 6, 7))

    dr.forward_to(y)
    assert dr.all(dr.grad(y) == m.ad.Float(30, 40, 60, 70))


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test18_diff_vjp_basic(t):
    # Spec 19.2 and 19.3: VJP with page-crossing and repeated indices
    pkg = get_pkg(t)
    m = sys.modules[t.__module__]
    UInt32 = m.UInt32

    dv = pkg.DiffPagedArrayViewF32([t(1, 2, 3), t(4, 5, 6), t(7, 8)], 8, 3)
    dv.enable_grad()

    y = pkg.gather_diff(dv, UInt32(2, 3, 5, 6))
    w = m.ad.Float(1, 2, 3, 4)
    dr.backward(dr.sum(y * w))
    assert dr.all(dv.grad() == t(0, 0, 1, 2, 0, 3, 4, 0))

    # Repeated indices accumulate
    dv.clear_grad()
    y = pkg.gather_diff(dv, UInt32(3, 3, 3))
    w = m.ad.Float(1, 2, 4)
    dr.backward(dr.sum(y * w))
    assert dr.all(dv.grad() == t(0, 0, 0, 7, 0, 0, 0, 0))


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test19_diff_shared_physical_page(t):
    # Spec 19.4: two logical ranges backed by the same physical page must
    # receive separate gradient entries
    pkg = get_pkg(t)
    m = sys.modules[t.__module__]
    UInt32 = m.UInt32

    page = t(5, 6, 7, 8)
    dv = pkg.DiffPagedArrayViewF32([page, page], 8, 4)
    dv.enable_grad()

    # Logical indices 1 (range one) and 5 (range two) read the same
    # physical value
    y = pkg.gather_diff(dv, UInt32(1, 5))
    assert dr.all(dr.detach(y) == m.ad.Float(6, 6))

    w = m.ad.Float(10, 20)
    dr.backward(dr.sum(y * w))

    # Gradients must not collapse despite identical physical pointers
    assert dr.all(dv.grad() == t(0, 10, 0, 0, 0, 20, 0, 0))


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test20_diff_lerp_page_boundary(t):
    # Spec 19.5: linear interpolation between the last element of one page
    # and the first element of the next, checked against finite differences
    pkg = get_pkg(t)
    m = sys.modules[t.__module__]
    UInt32 = m.UInt32

    values = [1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0]
    a = 0.375  # interpolation weight
    k = 3      # final element of page 0; k + 1 starts page 1

    def make(vals):
        return pkg.DiffPagedArrayViewF32([t(vals[0:4]), t(vals[4:8])], 8, 4)

    def lerp_result(dv):
        x0 = pkg.gather_diff(dv, UInt32(k))
        x1 = pkg.gather_diff(dv, UInt32(k + 1))
        return (1 - a) * x0 + a * x1

    # Reverse mode
    dv = make(values)
    dv.enable_grad()
    y = lerp_result(dv)
    dr.backward(y)
    g = dv.grad()
    expected = [0.0] * 8
    expected[k] = 1 - a
    expected[k + 1] = a
    assert dr.allclose(g, t(expected))

    # Forward mode: tangent picks out both interpolation partners
    dv2 = make(values)
    dv2.enable_grad()
    tangent = [0.0] * 8
    tangent[k] = 1.0
    tangent[k + 1] = 1.0
    dv2.set_grad(t(tangent))
    y2 = lerp_result(dv2)
    dr.forward_to(y2)
    assert dr.allclose(dr.grad(y2), 1.0)  # (1 - a) + a

    # Finite differences on the primal
    eps = 1e-2
    for i, w in ((k, 1 - a), (k + 1, a)):
        up = list(values)
        dn = list(values)
        up[i] += eps
        dn[i] -= eps
        fd = (dr.detach(lerp_result(make(up))) -
              dr.detach(lerp_result(make(dn)))) / (2 * eps)
        assert dr.allclose(fd, w, atol=1e-3)


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test21_diff_bilinear_page_boundary(t):
    # Spec 19.6: bilinear filtering with the four texels on multiple pages.
    # 4x4 texture, one row per page, so t00/t10 and t01/t11 live on
    # different pages.
    pkg = get_pkg(t)
    m = sys.modules[t.__module__]
    UInt32 = m.UInt32

    import random
    random.seed(5)
    tex = [float(random.randint(1, 100)) for _ in range(16)]

    x, y_ = 1, 2         # base texel
    fx, fy = 0.25, 0.625 # fractional position

    w00 = (1 - fx) * (1 - fy)
    w10 = fx * (1 - fy)
    w01 = (1 - fx) * fy
    w11 = fx * fy
    idx = {
        'i00': y_ * 4 + x,
        'i10': y_ * 4 + x + 1,
        'i01': (y_ + 1) * 4 + x,
        'i11': (y_ + 1) * 4 + x + 1,
    }

    def make(vals):
        pages = [t(vals[i:i + 4]) for i in range(0, 16, 4)]
        return pkg.DiffPagedArrayViewF32(pages, 16, 4)

    def sample(dv):
        t00 = pkg.gather_diff(dv, UInt32(idx['i00']))
        t10 = pkg.gather_diff(dv, UInt32(idx['i10']))
        t01 = pkg.gather_diff(dv, UInt32(idx['i01']))
        t11 = pkg.gather_diff(dv, UInt32(idx['i11']))
        return w00 * t00 + w10 * t10 + w01 * t01 + w11 * t11

    dv = make(tex)
    dv.enable_grad()
    r = sample(dv)
    dr.backward(r)
    g = dv.grad()

    expected = [0.0] * 16
    expected[idx['i00']] = w00
    expected[idx['i10']] = w10
    expected[idx['i01']] = w01
    expected[idx['i11']] = w11
    assert dr.allclose(g, t(expected))

    # Finite differences for all four texels
    eps = 1e-2
    for key, w in (('i00', w00), ('i10', w10), ('i01', w01), ('i11', w11)):
        up = list(tex)
        dn = list(tex)
        up[idx[key]] += eps
        dn[idx[key]] -= eps
        fd = (dr.detach(sample(make(up))) -
              dr.detach(sample(make(dn)))) / (2 * eps)
        assert dr.allclose(fd, w, atol=1e-2)


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test22_diff_trilinear_page_boundary(t):
    # Spec 19.7: trilinear interpolation with the eight voxels spread over
    # multiple pages (page size 3 never aligns with the 2x2x2 stencil)
    pkg = get_pkg(t)
    m = sys.modules[t.__module__]
    UInt32 = m.UInt32

    import random
    random.seed(7)
    grid = [float(random.randint(1, 50)) for _ in range(8)]  # 2x2x2

    fx, fy, fz = 0.25, 0.5, 0.75
    weights = []
    for z in (0, 1):
        for y_ in (0, 1):
            for x in (0, 1):
                w = ((fx if x else 1 - fx) *
                     (fy if y_ else 1 - fy) *
                     (fz if z else 1 - fz))
                weights.append(w)

    def make(vals):
        pages = [t(vals[0:3]), t(vals[3:6]), t(vals[6:8])]
        return pkg.DiffPagedArrayViewF32(pages, 8, 3)

    def sample(dv):
        r = None
        for i, w in enumerate(weights):
            v = w * pkg.gather_diff(dv, UInt32(i))
            r = v if r is None else r + v
        return r

    # Reverse mode
    dv = make(grid)
    dv.enable_grad()
    r = sample(dv)
    dr.backward(r)
    assert dr.allclose(dv.grad(), t(weights))

    # Forward mode: tangent of all ones gives the sum of the weights (1.0)
    dv2 = make(grid)
    dv2.enable_grad()
    dv2.set_grad(dr.full(t, 1.0, 8))
    r2 = sample(dv2)
    dr.forward_to(r2)
    assert dr.allclose(dr.grad(r2), 1.0)


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test23_diff_triangle_interpolation(t):
    # Spec 19.8: barycentric interpolation of three vertices stored on
    # three different pages
    pkg = get_pkg(t)
    m = sys.modules[t.__module__]
    UInt32 = m.UInt32

    positions = [float(i + 1) for i in range(9)]  # 9 scalar coordinates
    i0, i1, i2 = 0, 4, 8                          # one index per page
    b0, b1, b2 = 0.2, 0.3, 0.5

    dv = pkg.DiffPagedArrayViewF32(
        [t(positions[0:3]), t(positions[3:6]), t(positions[6:9])], 9, 3)
    dv.enable_grad()

    p0 = pkg.gather_diff(dv, UInt32(i0))
    p1 = pkg.gather_diff(dv, UInt32(i1))
    p2 = pkg.gather_diff(dv, UInt32(i2))
    p = b0 * p0 + b1 * p1 + b2 * p2

    dr.backward(p)
    expected = [0.0] * 9
    expected[i0] = b0
    expected[i1] = b1
    expected[i2] = b2
    assert dr.allclose(dv.grad(), t(expected))


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test24_diff_dot_identity(t):
    # Spec 19.9: dot(J v, w) == dot(v, J^T w) with indices crossing many
    # page boundaries
    pkg = get_pkg(t)
    m = sys.modules[t.__module__]
    UInt32 = m.UInt32

    import random
    random.seed(11)
    n, page_size = 23, 5
    values = [float(random.randint(1, 9)) for _ in range(n)]
    v = [float(random.randint(-5, 5)) for _ in range(n)]
    indices = [random.randrange(n) for _ in range(64)]
    w = [float(random.randint(-5, 5)) for _ in range(64)]

    pages = [t(values[i:i + page_size]) for i in range(0, n, page_size)]

    # J v (forward mode)
    dv = pkg.DiffPagedArrayViewF32(pages, n, page_size)
    dv.enable_grad()
    dv.set_grad(t(v))
    y = pkg.gather_diff(dv, UInt32(indices))
    dr.forward_to(y)
    lhs = dr.sum(dr.grad(y) * t(w))

    # J^T w (reverse mode)
    dv2 = pkg.DiffPagedArrayViewF32(pages, n, page_size)
    dv2.enable_grad()
    y2 = pkg.gather_diff(dv2, UInt32(indices))
    dr.backward(dr.sum(y2 * m.ad.Float(w)))
    rhs = dr.sum(dv2.grad() * t(v))

    assert dr.allclose(lhs, rhs)


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test25_diff_snapshot_replacement(t):
    # Spec 19.10: forward uses snapshot A; the pages are then replaced; the
    # backward pass of the old result must correspond to snapshot A
    pkg = get_pkg(t)
    m = sys.modules[t.__module__]
    UInt32 = m.UInt32

    dv = pkg.DiffPagedArrayViewF32([t(1, 2, 3, 4), t(5, 6, 7, 8)], 8, 4)
    dv.enable_grad()

    idx = UInt32(3, 4)
    y = pkg.gather_diff(dv, idx)  # unevaluated, uses snapshot A

    # Replace both pages (snapshot B)
    dv.update_page(0, t(10, 20, 30, 40))
    dv.update_page(1, t(50, 60, 70, 80))

    # The old expression still reads snapshot A
    dr.backward(dr.sum(y * m.ad.Float(2, 3)))
    assert dr.all(dr.detach(y) == m.ad.Float(4, 5))

    # Gradients route to the logical indices (page-independent)
    assert dr.all(dv.grad() == t(0, 0, 0, 2, 3, 0, 0, 0))

    # A new gather reads snapshot B
    y2 = pkg.gather_diff(dv, idx)
    assert dr.all(dr.detach(y2) == m.ad.Float(40, 50))


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test26_diff_freeze_replay(t):
    # Spec 21: frozen primal, JVP, and VJP must replay after page
    # replacement, reading the new values without retracing
    pkg = get_pkg(t)
    m = sys.modules[t.__module__]
    UInt32 = m.UInt32

    idx = dr.arange(UInt32, 8)

    # Primal
    holder = pkg.DiffPagedHolderF32([t(1, 2, 3, 4), t(5, 6, 7, 8)], 8, 4)

    @dr.freeze
    def primal(holder, idx):
        return pkg.gather_diff_holder(holder, idx)

    r1 = primal(holder, idx)
    assert dr.all(dr.detach(r1) == m.ad.Float(1, 2, 3, 4, 5, 6, 7, 8))
    holder.update_page(0, t(10, 20, 30, 40))
    r2 = primal(holder, idx)
    assert dr.all(dr.detach(r2) == m.ad.Float(10, 20, 30, 40, 5, 6, 7, 8))
    assert primal.n_recordings == 1

    # VJP
    holder2 = pkg.DiffPagedHolderF32([t(1, 2, 3, 4), t(5, 6, 7, 8)], 8, 4)
    holder2.enable_grad()

    @dr.freeze
    def vjp(holder, idx, w):
        holder.clear_grad()
        y = pkg.gather_diff_holder(holder, idx)
        dr.backward(dr.sum(y * w))
        return holder.grad()

    w = m.ad.Float([1, 2, 3, 4, 5, 6, 7, 8])
    g1 = vjp(holder2, idx, w)
    assert dr.all(g1 == t(1, 2, 3, 4, 5, 6, 7, 8))
    holder2.update_page(1, t(50, 60, 70, 80))
    # Reset the proxy gradient so that the AD state at entry matches the
    # first call (a different entry state would legitimately retrace)
    holder2.clear_grad()
    g2 = vjp(holder2, idx, w)
    assert dr.all(g2 == t(1, 2, 3, 4, 5, 6, 7, 8))
    assert vjp.n_recordings == 1

    # JVP
    holder3 = pkg.DiffPagedHolderF32([t(1, 2, 3, 4), t(5, 6, 7, 8)], 8, 4)
    holder3.enable_grad()

    @dr.freeze
    def jvp(holder, idx, tangent):
        holder.clear_grad()
        holder.set_grad(tangent)
        y = pkg.gather_diff_holder(holder, idx)
        dr.forward_to(y)
        return dr.grad(y)

    tangent = t([10, 20, 30, 40, 50, 60, 70, 80])
    d1 = jvp(holder3, idx, tangent)
    assert dr.all(d1 == m.ad.Float(10, 20, 30, 40, 50, 60, 70, 80))
    holder3.update_page(0, t(-1, -2, -3, -4))
    holder3.clear_grad()
    d2 = jvp(holder3, idx, tangent)
    assert dr.all(d2 == m.ad.Float(10, 20, 30, 40, 50, 60, 70, 80))
    assert jvp.n_recordings == 1


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test27_scalar_index_snapshot(t):
    # Regression test: gathers with a scalar literal index previously lost
    # the dependency on the pointer table (and thereby on the page
    # retainer) through a size-1 gather elision in the JIT core. The pages
    # of a destroyed view could then be reused before evaluation.
    pkg = get_pkg(t)
    UInt32 = sys.modules[t.__module__].UInt32
    import gc

    def make(vals):
        return pkg.PagedArrayViewF32([t(vals[0:4]), t(vals[4:8])], 8, 4)

    values = [1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0]

    # The views are temporaries: only the unevaluated expressions survive
    x0 = pkg.gather(make(values), UInt32(3))
    x1 = pkg.gather(make(values), UInt32(4))
    gc.collect()

    # Try to provoke reuse of any prematurely freed page/table storage
    for _ in range(8):
        dr.eval(dr.full(t, 999.0, 4))

    dr.eval(x0, x1)
    assert dr.all(x0 == t(4)) and dr.all(x1 == t(10))
