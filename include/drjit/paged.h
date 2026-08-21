/*
    drjit/paged.h -- Read-only view of an array split into separately
    allocated fixed-capacity pages, accessed through the ordinary
    dr::gather() interface.

    A PagedArrayView presents a sequence of independently allocated arrays
    ("pages") as one logical array. All pages except the last must have
    exactly 'page_size' entries; the final page may be partial. Reads
    perform a two-level indirection: a first gather fetches the relevant
    page pointer from a small table, and a second (indirect) gather then
    dereferences it. The generated kernel is therefore independent of the
    page count, and pages can be shared or replaced between kernel launches
    without recompilation.

    Example:

        using Float = dr::CUDAArray<float>;

        dr::PagedArrayView<Float> positions_x(pages, page_count,
                                              logical_size, page_size);

        Float x = dr::gather<Float>(positions_x, index, active);

    Lifetime rules: the view owns references to the page arrays, and every
    pointer table additionally retains the pages whose addresses it stores.
    Each table therefore represents an immutable storage snapshot, and
    unevaluated gather expressions remain valid even if the view is
    destroyed or pages are replaced before evaluation. However, Dr.Jit
    cannot track reads through raw pointers at the kernel level: page
    contents must not be mutated in place while a scheduled computation
    that reads through the view may still execute; synchronize first. For
    pages created via drjit::JitArray::map_() from externally owned memory,
    the caller must additionally keep the external allocation alive.

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit/jit.h>
#include <drjit/autodiff.h>
#include <memory>
#include <vector>

NAMESPACE_BEGIN(drjit)

NAMESPACE_BEGIN(detail)

/**
 * \brief Ownership payload attached to a pointer-table variable
 *
 * A pointer table stores raw addresses, which do not constitute a JIT
 * dependency on the page variables whose storage they reference. This
 * payload holds references to those pages and releases them when the table
 * variable itself is freed. Expressions that read through the table
 * therefore keep the referenced pages alive even if the originating
 * PagedArrayView is destroyed or repopulated in the meantime.
 */
struct paged_table_retainer {
    std::vector<uint32_t> indices;

    ~paged_table_retainer() {
        for (uint32_t index : indices)
            jit_var_dec_ref(index);
    }

    static void callback(uint32_t /* index */, int free, void *payload) {
        if (free)
            delete (paged_table_retainer *) payload;
    }
};

/**
 * \brief Internal accessor granting mutable access to the JIT variables of
 * a PagedArrayView (see below)
 *
 * This is exclusively meant for object traversal glue (e.g. to support
 * function freezing, see drjit/traversable_base.h), which substitutes
 * variables 1:1 and hence preserves the correspondence between page
 * storage and pointer-table entries. It is not part of the public API;
 * structural changes must go through PagedArrayView::update_page() and
 * PagedArrayView::rebuild_page_table().
 */
template <typename Array> struct paged_array_view_access {
    using View = PagedArrayView<Array>;

    static std::vector<Array> &pages(View &view) { return view.m_pages; }

    static typename View::Pointer &page_table(View &view) {
        return view.m_page_table;
    }
};

/// Equivalent of the above for DiffPagedArrayView (see below)
template <typename Array> struct diff_paged_array_view_access {
    using View = DiffPagedArrayView<Array>;

    static typename View::Primal &primal(View &view) { return view.m_primal; }
    static Array &proxy(View &view) { return view.m_proxy; }
};

NAMESPACE_END(detail)

template <typename Array_> class PagedArrayView {
public:
    using Array   = Array_;
    using Value   = value_t<Array>;
    using Index   = uint32_array_t<Array>;
    using Mask    = mask_t<Array>;
    using Pointer = JitArray<backend_v<Array>, void *>;

    static_assert(is_jit_v<Array> && depth_v<Array> == 1 && !is_tensor_v<Array>,
                  "PagedArrayView<T>: 'T' must be a flat JIT-compiled array "
                  "type (e.g. drjit::CUDAArray<float>)!");

    static_assert(!is_diff_v<Array>,
                  "Differentiable PagedArrayView is not implemented yet. "
                  "Construct the view over detached (non-AD) arrays if this "
                  "is intended.");

    PagedArrayView() = default;

    /**
     * \brief Create a paged view of ``logical_size`` entries backed by
     * ``page_count`` pages with a capacity of ``page_size`` entries each
     *
     * All pages except the last must have exactly ``page_size`` entries;
     * the final page must hold the remainder. The constructor evaluates
     * every page and builds the pointer table. The page handles are copied
     * into the view, which keeps the underlying storage referenced.
     */
    PagedArrayView(const Array *pages, size_t page_count, size_t logical_size,
                   size_t page_size) {
        if (logical_size == 0) {
            if (page_count != 0)
                jit_raise("PagedArrayView(): an empty view cannot have pages!");
            return;
        }

        if (page_size == 0)
            jit_raise("PagedArrayView(): 'page_size' must be nonzero!");

        if (logical_size > 0xFFFFFFFFull)
            jit_raise("PagedArrayView(): logical sizes above 2^32 - 1 are "
                      "not supported!");

        if (page_size > 0xFFFFFFFFull)
            jit_raise("PagedArrayView(): page sizes above 2^32 - 1 are "
                      "not supported!");

        size_t expected =
            logical_size / page_size + (logical_size % page_size != 0);
        if (page_count != expected)
            jit_raise("PagedArrayView(): a view with %zu entries and a page "
                      "size of %zu requires %zu pages (%zu were provided)!",
                      logical_size, page_size, expected, page_count);

        m_size = logical_size;
        m_page_size = page_size;
        m_power_of_two = (page_size & (page_size - 1)) == 0;

        if (m_power_of_two) {
            uint32_t shift = 0;
            while ((page_size >> shift) > 1)
                shift++;
            m_page_shift = shift;
            m_page_mask = (uint32_t) (page_size - 1);
        }

        m_pages.assign(pages, pages + page_count);
        rebuild_page_table();
    }

    PagedArrayView(const std::vector<Array> &pages, size_t logical_size,
                   size_t page_size)
        : PagedArrayView(pages.data(), pages.size(), logical_size,
                         page_size) { }

    /// Number of entries of the logical array
    size_t size() const { return m_size; }

    /// Capacity of a single page, in entries
    size_t page_size() const { return m_page_size; }

    /// Number of pages
    size_t page_count() const { return m_pages.size(); }

    /// Is the page size a power of two? (enables cheaper addressing)
    bool power_of_two_page_size() const { return m_power_of_two; }

    /// Pointer table with one entry per page
    const Pointer &page_table() const { return m_page_table; }

    /// Page handles owned by this view
    const std::vector<Array> &pages() const { return m_pages; }

    /// Number of entries expected of page ``i``
    size_t page_length(size_t i) const {
        if (i >= m_pages.size())
            jit_raise("PagedArrayView::page_length(): page index %zu is out "
                      "of bounds (the view has %zu pages)!",
                      i, m_pages.size());

        return (i + 1 < m_pages.size())
                   ? m_page_size
                   : m_size - (m_pages.size() - 1) * m_page_size;
    }

    /**
     * \brief Replace a single page
     *
     * The new page must match the size of the page that it replaces. The
     * pointer table is rebuilt. Previously created gather expressions
     * retain the old pointer table, which in turn retains the old page
     * storage, so replacing a page does not require synchronization.
     * In-place mutation or destruction of externally owned mapped storage
     * still requires the caller to synchronize outstanding computations.
     */
    void update_page(size_t page_index, const Array &new_page) {
        if (page_index >= m_pages.size())
            jit_raise("PagedArrayView::update_page(): page index %zu is out "
                      "of bounds (the view has %zu pages)!",
                      page_index, m_pages.size());

        size_t expected = page_length(page_index);
        if (new_page.size() != expected)
            jit_raise("PagedArrayView::update_page(): page %zu has %zu "
                      "entries, but %zu were expected!",
                      page_index, new_page.size(), expected);

        m_pages[page_index] = new_page;
        rebuild_page_table();
    }

    /**
     * \brief Recreate the pointer table from the current set of pages
     *
     * This evaluates every page handle owned by the view, validates its
     * size, and uploads a fresh table of page addresses. It only needs to
     * be called directly when the storage of an owned page handle was
     * re-materialized (e.g. by an explicit migration); adopting replacement
     * pages should instead go through \ref update_page(), which rebuilds
     * the table automatically.
     *
     * The created table variable retains references to the pages whose
     * addresses it stores. Unevaluated gather expressions therefore remain
     * valid even if the view is destroyed or its pages are replaced before
     * evaluation; each table represents an immutable storage snapshot.
     *
     * Note that rebuilding only affects gathers issued afterwards. It
     * cannot retroactively repair older pointer tables whose raw addresses
     * were invalidated because storage behind them was moved or freed by
     * an external owner.
     */
    void rebuild_page_table() {
        size_t page_count = m_pages.size();
        if (page_count == 0) {
            m_page_table = Pointer();
            return;
        }

        std::unique_ptr<void *[]> table(new void *[page_count]);
        std::unique_ptr<detail::paged_table_retainer> retainer(
            new detail::paged_table_retainer());
        retainer->indices.reserve(page_count);

        for (size_t i = 0; i < page_count; ++i) {
            const Array &page = m_pages[i];
            size_t expected = page_length(i);

            if (page.size() != expected)
                jit_raise("PagedArrayView(): page %zu has %zu entries, but "
                          "%zu were expected!", i, page.size(), expected);

            /* Evaluate the page and fetch the address of its storage */
            void *ptr = (void *) page.data();
            if (!ptr)
                jit_raise("PagedArrayView(): page %zu does not expose a "
                          "storage address!", i);

            table[i] = ptr;
            jit_var_inc_ref(page.index());
            retainer->indices.push_back(page.index());
        }

        m_page_table = Pointer::load_(table.get(), page_count);

        /* Tie the lifetime of the referenced pages to the table variable */
        jit_var_set_callback(m_page_table.index(),
                             detail::paged_table_retainer::callback,
                             retainer.release());
    }

    /// Implementation of dr::gather() for paged sources (see array_router.h)
    template <typename Target, typename Index2, typename Mask2>
    Target gather_(const Index2 &index_, const Mask2 &mask_,
                   ReduceMode mode) const {
        if constexpr (depth_v<Target> > 1) {
            /* Nested target (e.g. gather<Array3f>): the packet of a lane
               may straddle a page boundary, hence gather each entry
               separately. */
            static_assert(size_v<Target> != Dynamic,
                          "gather(): dynamically sized target types are not "
                          "supported when gathering from a PagedArrayView!");

            using Entry = value_t<Target>;
            constexpr uint32_t N = (uint32_t) size_v<Target>;

            Index index(index_);
            Target result;
            for (uint32_t i = 0; i < N; ++i)
                result.entry(i) = gather_<Entry>(
                    fmadd(index, Index(N), Index(i)), mask_, mode);
            return result;
        } else {
            static_assert(std::is_same_v<Target, Array>,
                          "gather(): the target type must match the array "
                          "type underlying the PagedArrayView!");

            if (m_size == 0)
                return zeros<Target>(width(index_));

            Index index(index_);
            Mask valid = Mask(mask_) && (index < Index((uint32_t) m_size));

            /* A single-page view is contiguous, and the page-table
               indirection must be skipped rather than merely wasted: a
               gather from a size-1 pointer table is elided by
               jitc_var_gather() into and(pointer, mask), whose LLVM
               lowering emits a 'select' typed from the type table (i64)
               against a pointer variable defined as an opaque 'ptr'
               vector -- the kernel then fails to parse and Dr.Jit shuts
               the process down. Routing it through a real gather node
               instead is no better: a pointer-typed gather result
               crossing a symbolic call boundary is double-emitted by the
               call machinery. Gathering from the page directly sidesteps
               the pointer round trip entirely and is the ordinary,
               well-exercised data gather. */
            if (m_pages.size() == 1)
                return drjit::gather<Target>(m_pages[0], index, valid, mode);

            Index page, offset;
            if (m_power_of_two) {
                page   = index >> m_page_shift;
                offset = index & Index(m_page_mask);
            } else {
                page   = index / Index((uint32_t) m_page_size);
                offset = fnmadd(page, Index((uint32_t) m_page_size), index);
            }

            /* Out-of-bounds lanes are disabled in 'valid' and gather a null
               pointer here, which the indirect gather below never
               dereferences. */
            Pointer ptr = drjit::gather<Pointer>(m_page_table, page, valid, mode);

            return Target::gather_ptr_(ptr, offset, valid);
        }
    }

private:
    /// Traversal glue with mutable access (not part of the public API)
    friend struct detail::paged_array_view_access<Array_>;

    /// Owned page handles
    std::vector<Array> m_pages;

    /// Device-visible table with one pointer per page
    Pointer m_page_table;

    /// Number of entries of the logical array
    size_t m_size = 0;

    /// Capacity of a single page, in entries
    size_t m_page_size = 0;

    /// Fast-path bookkeeping when 'm_page_size' is a power of two
    bool m_power_of_two = false;
    uint32_t m_page_shift = 0;
    uint32_t m_page_mask = 0;
};

/**
 * \brief Differentiable view of an array with paged primal storage
 *
 * DiffPagedArrayView augments a read-only PagedArrayView with a single
 * *logical* AD identity: from the perspective of automatic differentiation,
 * the paged input is one differentiable array of ``logical_size`` entries.
 * The pages are only how this array is read during primal execution; page
 * boundaries and physical page sharing have no differentiation semantics.
 *
 * Derivatives are carried by an internal *AD proxy*, a differentiable array
 * of the logical width whose primal contents are never read (they remain a
 * literal zero). Reads via dr::gather() combine
 *
 * - a primal gather through the paged storage, and
 * - the AD edge of an ordinary gather from the proxy.
 *
 * The resulting derivatives are those of a contiguous logical array:
 *
 * - forward mode gathers from the logical tangent by logical index;
 * - reverse mode scatter-adds adjoints into the logical gradient by
 *   logical index (repeated indices accumulate).
 *
 * Consequently, gradients are indexed by logical position: two logical
 * ranges that share the same physical page still receive separate gradient
 * entries, and interpolation across a page boundary differentiates exactly
 * like any other pair of adjacent indices.
 *
 * Typical use:
 *
 *     DiffPagedArrayView<Float> values(pages, page_count, n, page_size);
 *     values.enable_grad();
 *
 *     Float y = dr::gather<Float>(values, index, active);
 *     dr::backward(loss(y));
 *
 *     auto dx = values.gradient(); // one logical gradient of width n
 *
 * IMPORTANT SEMANTIC NOTES
 *
 * 1. The AD proxy is a *derivative carrier*, not a mutable parameter:
 *    updating its primal values has no effect on the paged primal storage.
 *    The host applies parameter updates to its own authoritative storage
 *    and constructs a new view (a new snapshot) for the next operation.
 *
 * 2. A DiffPagedArrayView represents one immutable snapshot bound to one
 *    AD identity. The type therefore exposes no page mutation: replacing
 *    logical page values requires constructing a new view (with a fresh
 *    proxy). Mixing reads of different snapshots through one AD variable
 *    would silently blend derivatives evaluated at different points.
 *    Compiled kernels are still reused across snapshots, since page
 *    addresses remain runtime inputs.
 *
 * 3. When integrating with function freezing (dr.freeze), recordings must
 *    be keyed on the paging *structure*: generated kernels bake the
 *    bounds and page/offset arithmetic derived from logical_size and
 *    page_size, while page addresses and contents remain runtime inputs.
 *    Traversable wrappers should reject structural changes on snapshot
 *    rebinding, and additionally contribute structure-dependent variables
 *    to the traversed layout (e.g. literal zeros whose widths encode the
 *    page size and the final page length; variable widths are always part
 *    of the frozen input layout). See the holders in tests/paged_ext.cpp
 *    for the reference pattern.
 *
 * Tangents and gradients use contiguous storage of the logical width; only
 * the primal is paged. Derivative propagation never reads the pages, so
 * gradients always correspond to the snapshot that produced the primal.
 */
template <typename Array_> class DiffPagedArrayView {
public:
    using Array    = Array_;
    using Detached = detached_t<Array>;
    using Primal   = PagedArrayView<Detached>;
    using Index    = uint32_array_t<Detached>;
    using Mask     = mask_t<Detached>;

    static_assert(is_diff_v<Array> && depth_v<Array> == 1 && !is_tensor_v<Array>,
                  "DiffPagedArrayView<T>: 'T' must be a flat differentiable "
                  "array type (e.g. drjit::DiffArray<JitBackend::CUDA, float>)!");

    DiffPagedArrayView() = default;

    /// Wrap an existing primal view (snapshot)
    explicit DiffPagedArrayView(const Primal &primal)
        : m_primal(primal), m_proxy(make_proxy(primal.size())) { }

    DiffPagedArrayView(const Detached *pages, size_t page_count,
                       size_t logical_size, size_t page_size)
        : m_primal(pages, page_count, logical_size, page_size),
          m_proxy(make_proxy(logical_size)) { }

    DiffPagedArrayView(const std::vector<Detached> &pages, size_t logical_size,
                       size_t page_size)
        : m_primal(pages, logical_size, page_size),
          m_proxy(make_proxy(logical_size)) { }

    /// Number of entries of the logical array
    size_t size() const { return m_primal.size(); }

    /// The underlying read-only paged snapshot
    const Primal &primal() const { return m_primal; }

    /// Enable derivative tracking for this view
    void enable_grad() { drjit::enable_grad(m_proxy); }

    /// Disable derivative tracking for this view
    void disable_grad() { drjit::disable_grad(m_proxy); }

    /// Is derivative tracking enabled?
    bool grad_enabled() const { return drjit::grad_enabled(m_proxy); }

    /// Set the logical tangent for forward-mode differentiation
    void set_tangent(const Detached &value) {
        drjit::set_grad(m_proxy, value);
    }

    /// The logical tangent (forward mode)
    Detached tangent() const { return drjit::grad<false>(m_proxy); }

    /// The accumulated logical gradient (reverse mode)
    Detached gradient() const { return drjit::grad<false>(m_proxy); }

    /// Reset tangent/gradient state
    void clear_gradient() { drjit::clear_grad(m_proxy); }

    /**
     * \brief The logical AD identity of this view
     *
     * This array only carries tangent and adjoint state; its primal
     * contents are unused, and updating them has no effect on the paged
     * primal storage. Exposed for traversal/integration glue; typical
     * callers should use enable_grad(), set_tangent(), gradient(), and
     * clear_gradient() instead.
     */
    const Array &ad_proxy() const { return m_proxy; }

    /// Implementation of dr::gather() for differentiable paged sources
    template <typename Target, typename Index2, typename Mask2>
    Target gather_(const Index2 &index_, const Mask2 &mask_,
                   ReduceMode mode) const {
        if constexpr (depth_v<Target> > 1) {
            static_assert(size_v<Target> != Dynamic,
                          "gather(): dynamically sized target types are not "
                          "supported when gathering from a "
                          "DiffPagedArrayView!");

            using Entry = value_t<Target>;
            constexpr uint32_t N = (uint32_t) size_v<Target>;

            Index index(detach<false>(index_));
            Target result;
            for (uint32_t i = 0; i < N; ++i)
                result.entry(i) = gather_<Entry>(
                    fmadd(index, Index(N), Index(i)), mask_, mode);
            return result;
        } else {
            static_assert(std::is_same_v<Target, Array>,
                          "gather(): the target type must match the array "
                          "type underlying the DiffPagedArrayView!");

            /* Indices and masks are not differentiable quantities */
            Index index = Index(detach<false>(index_));
            Mask mask   = Mask(detach<false>(mask_));

            /* Out-of-range lanes are disabled for the primal read and the
               derivative edge alike: they yield zero in the primal, a zero
               forward derivative, and no reverse-mode contribution. */
            Mask valid = mask && (index < Index((uint32_t) m_primal.size()));

            /* Primal read through the paged storage */
            Detached primal =
                drjit::gather<Detached>(m_primal, index, valid, mode);

            if (!drjit::grad_enabled(m_proxy))
                return Target(primal);

            /* Differentiable read of the AD proxy. Its primal content (a
               literal zero) is discarded by replace_grad() below; the
               operation only contributes the AD edge, whose forward
               derivative gathers from the logical tangent, and whose
               reverse derivative scatter-adds into the logical gradient. */
            using DiffIndex = uint32_array_t<Array>;
            using DiffMask  = mask_t<Array>;
            Target wired = drjit::gather<Target>(m_proxy, DiffIndex(index),
                                                 DiffMask(valid), mode);

            return replace_grad(Target(primal), wired);
        }
    }

private:
    /// Traversal glue with mutable access (not part of the public API)
    friend struct detail::diff_paged_array_view_access<Array_>;

    static Array make_proxy(size_t size) {
        return size ? zeros<Array>(size) : Array();
    }

    /// Read-only paged primal storage
    Primal m_primal;

    /// One logical AD identity of the logical width
    Array m_proxy;
};

// -----------------------------------------------------------------------
//  Adapter overloads so that generic code can treat paged views like
//  ordinary flat arrays where this is meaningful
// -----------------------------------------------------------------------

template <typename T> size_t width(const PagedArrayView<T> &view) {
    return view.size();
}

template <typename T> size_t width(const DiffPagedArrayView<T> &view) {
    return view.size();
}

template <typename T> bool grad_enabled(const DiffPagedArrayView<T> &view) {
    return view.grad_enabled();
}

template <typename T> void enable_grad(DiffPagedArrayView<T> &view) {
    view.enable_grad();
}

template <typename T> void disable_grad(DiffPagedArrayView<T> &view) {
    view.disable_grad();
}

template <typename T>
typename DiffPagedArrayView<T>::Detached
grad(const DiffPagedArrayView<T> &view) {
    return view.gradient();
}

template <typename T, typename T2>
void set_grad(DiffPagedArrayView<T> &view, const T2 &value) {
    view.set_tangent(value);
}

template <typename T> void clear_grad(DiffPagedArrayView<T> &view) {
    view.clear_gradient();
}

NAMESPACE_END(drjit)
