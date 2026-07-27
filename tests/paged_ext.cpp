/* Note: drjit/paged.h is deliberately the first Dr.Jit include, which
   verifies that the header is self-contained (including for the
   differentiable view). */
#include <drjit/paged.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <drjit/python.h>
#include <drjit/traversable_base.h>

namespace nb = nanobind;
namespace dr = drjit;

using namespace nb::literals;

/**
 * Traversable wrapper around PagedArrayView. Only the pointer table is
 * exposed to Dr.Jit's object traversal mechanism: it is the only JIT
 * variable that generated kernels access (and it retains the pages), so
 * frozen-function preparation stays O(1) in the page count.
 */
template <JitBackend Backend>
class PagedHolder : public dr::TraversableBase {
public:
    using Float = dr::JitArray<Backend, float>;

    PagedHolder() = default;
    PagedHolder(const std::vector<Float> &pages, size_t logical_size,
                size_t page_size)
        : m_view(pages, logical_size, page_size) { }

    dr::PagedArrayView<Float> &view() { return m_view; }
    const dr::PagedArrayView<Float> &view() const { return m_view; }

    void traverse_1_cb_ro(void *payload,
                          dr::detail::traverse_callback_ro fn) const override {
        dr::traverse_1_fn_ro(m_view.page_table(), payload, fn);
    }

    void traverse_1_cb_rw(void *payload,
                          dr::detail::traverse_callback_rw fn) override {
        using Access = dr::detail::paged_array_view_access<Float>;
        dr::traverse_1_fn_rw(Access::page_table(m_view), payload, fn);
    }

private:
    dr::PagedArrayView<Float> m_view;
};

/**
 * Traversable wrapper around DiffPagedArrayView. Exposes the pointer table
 * (primal) and the AD proxy (derivatives) to object traversal; see above.
 * Page value changes go through rebind(), which installs a fresh snapshot
 * with a fresh AD identity, following the one-snapshot-per-AD-identity
 * rule of DiffPagedArrayView.
 */
template <JitBackend Backend>
class DiffPagedHolder : public dr::TraversableBase {
public:
    using Float     = dr::JitArray<Backend, float>;
    using DiffFloat = dr::DiffArray<Backend, float>;
    using View      = dr::DiffPagedArrayView<DiffFloat>;

    DiffPagedHolder() = default;
    DiffPagedHolder(const std::vector<Float> &pages, size_t logical_size,
                    size_t page_size)
        : m_view(pages, logical_size, page_size) { }

    View &view() { return m_view; }
    const View &view() const { return m_view; }

    /// Install a new snapshot (fresh pages, fresh AD identity)
    void rebind(const std::vector<Float> &pages, size_t logical_size,
                size_t page_size) {
        bool tracked = m_view.grad_enabled();
        m_view = View(pages, logical_size, page_size);
        if (tracked)
            m_view.enable_grad();
    }

    void traverse_1_cb_ro(void *payload,
                          dr::detail::traverse_callback_ro fn) const override {
        dr::traverse_1_fn_ro(m_view.primal().page_table(), payload, fn);
        dr::traverse_1_fn_ro(m_view.ad_proxy(), payload, fn);
    }

    void traverse_1_cb_rw(void *payload,
                          dr::detail::traverse_callback_rw fn) override {
        using PAccess = dr::detail::paged_array_view_access<Float>;
        using DAccess = dr::detail::diff_paged_array_view_access<DiffFloat>;
        dr::traverse_1_fn_rw(PAccess::page_table(DAccess::primal(m_view)),
                             payload, fn);
        dr::traverse_1_fn_rw(DAccess::proxy(m_view), payload, fn);
    }

private:
    View m_view;
};

/// Bindings shared between the float32 and float64 variants
template <JitBackend Backend, typename Value>
void bind_typed(nb::module_ m, const char *suffix) {
    using Plain     = dr::JitArray<Backend, Value>;
    using UInt32    = dr::JitArray<Backend, uint32_t>;
    using Mask      = dr::mask_t<Plain>;
    using View      = dr::PagedArrayView<Plain>;
    using DiffType  = dr::DiffArray<Backend, Value>;
    using DiffView  = dr::DiffPagedArrayView<DiffType>;

    nb::class_<View>(m, (std::string("PagedArrayView") + suffix).c_str())
        .def(nb::init<>())
        .def(nb::init<const std::vector<Plain> &, size_t, size_t>(),
             "pages"_a, "logical_size"_a, "page_size"_a)
        .def("size", &View::size)
        .def("page_size", &View::page_size)
        .def("page_count", &View::page_count)
        .def("power_of_two_page_size", &View::power_of_two_page_size)
        .def("page_length", &View::page_length)
        .def("update_page", &View::update_page)
        .def("rebuild_page_table", &View::rebuild_page_table);

    m.def("gather", [](const View &view, const UInt32 &index, const Mask &active) {
        return dr::gather<Plain>(view, index, active);
    }, "view"_a, "index"_a, "active"_a);

    m.def("gather", [](const View &view, const UInt32 &index) {
        return dr::gather<Plain>(view, index);
    }, "view"_a, "index"_a);

    nb::class_<DiffView>(m, (std::string("DiffPagedArrayView") + suffix).c_str())
        .def(nb::init<>())
        .def(nb::init<const std::vector<Plain> &, size_t, size_t>(),
             "pages"_a, "logical_size"_a, "page_size"_a)
        .def("size", &DiffView::size)
        .def("enable_grad", &DiffView::enable_grad)
        .def("grad_enabled", &DiffView::grad_enabled)
        .def("set_tangent", &DiffView::set_tangent)
        .def("tangent", &DiffView::tangent)
        .def("gradient", &DiffView::gradient)
        .def("clear_gradient", &DiffView::clear_gradient);

    m.def("gather_diff", [](const DiffView &view, const UInt32 &index,
                            const Mask &active) {
        return dr::gather<DiffType>(view, index, active);
    }, "view"_a, "index"_a, "active"_a);

    m.def("gather_diff", [](const DiffView &view, const UInt32 &index) {
        return dr::gather<DiffType>(view, index);
    }, "view"_a, "index"_a);
}

template <JitBackend Backend> void bind(nb::module_ m) {
    using Float  = dr::JitArray<Backend, float>;
    using UInt32 = dr::JitArray<Backend, uint32_t>;
    using Mask   = dr::mask_t<Float>;
    using Float3 = dr::Array<Float, 3>;
    using View   = dr::PagedArrayView<Float>;
    using Holder = PagedHolder<Backend>;
    using DiffFloat  = dr::DiffArray<Backend, float>;
    using DiffHolder = DiffPagedHolder<Backend>;

    bind_typed<Backend, float>(m, "F32");
    bind_typed<Backend, double>(m, "F64");

    m.def("gather3", [](const View &view, const UInt32 &index, const Mask &active) {
        Float3 r = dr::gather<Float3>(view, index, active);
        return std::make_tuple(r.x(), r.y(), r.z());
    }, "view"_a, "index"_a, "active"_a);

    m.def("jit_ref_count", [](const Float &value) {
        return jit_var_ref(value.index());
    });

    auto holder = nb::class_<Holder>(
        m, "PagedHolderF32",
        nb::intrusive_ptr<Holder>(
            [](Holder *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def(nb::init<>())
        .def(nb::init<const std::vector<Float> &, size_t, size_t>(),
             "pages"_a, "logical_size"_a, "page_size"_a)
        .def("size", [](const Holder &h) { return h.view().size(); })
        .def("update_page", [](Holder &h, size_t i, const Float &page) {
            h.view().update_page(i, page);
        });

    dr::bind_traverse(holder);

    m.def("gather_holder", [](const Holder &holder, const UInt32 &index) {
        return dr::gather<Float>(holder.view(), index);
    }, "holder"_a, "index"_a);

    auto diff_holder = nb::class_<DiffHolder>(
        m, "DiffPagedHolderF32",
        nb::intrusive_ptr<DiffHolder>(
            [](DiffHolder *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def(nb::init<>())
        .def(nb::init<const std::vector<Float> &, size_t, size_t>(),
             "pages"_a, "logical_size"_a, "page_size"_a)
        .def("size", [](const DiffHolder &h) { return h.view().size(); })
        .def("rebind", &DiffHolder::rebind,
             "pages"_a, "logical_size"_a, "page_size"_a)
        .def("enable_grad", [](DiffHolder &h) { h.view().enable_grad(); })
        .def("set_tangent", [](DiffHolder &h, const Float &value) {
            h.view().set_tangent(value);
        })
        .def("gradient", [](DiffHolder &h) { return h.view().gradient(); })
        .def("clear_gradient", [](DiffHolder &h) {
            h.view().clear_gradient();
        });

    dr::bind_traverse(diff_holder);

    m.def("gather_diff_holder", [](const DiffHolder &holder, const UInt32 &index) {
        return dr::gather<DiffFloat>(holder.view(), index);
    }, "holder"_a, "index"_a);
}

NB_MODULE(paged_ext, m) {
    nb::module_::import_("drjit");

    nb::intrusive_init(
        [](PyObject *o) noexcept {
            nb::gil_scoped_acquire guard;
            Py_INCREF(o);
        },
        [](PyObject *o) noexcept {
            nb::gil_scoped_acquire guard;
            Py_DECREF(o);
        });

#if defined(DRJIT_ENABLE_LLVM)
    bind<JitBackend::LLVM>(m.def_submodule("llvm"));
#endif

#if defined(DRJIT_ENABLE_CUDA)
    bind<JitBackend::CUDA>(m.def_submodule("cuda"));
#endif

#if defined(DRJIT_ENABLE_METAL)
    bind<JitBackend::Metal>(m.def_submodule("metal"));
#endif
}
