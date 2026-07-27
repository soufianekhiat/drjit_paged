#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <drjit/python.h>
#include <drjit/autodiff.h>
#include <drjit/paged.h>
#include <drjit/traversable_base.h>

namespace nb = nanobind;
namespace dr = drjit;

using namespace nb::literals;

/**
 * Traversable wrapper around PagedArrayView. Exposing the view's JIT
 * variables (pages and pointer table) to Dr.Jit's object traversal
 * mechanism makes the holder usable as an input of frozen functions:
 * replaying with an updated view rebinds the pointer table rather than
 * baking stale addresses into the recording.
 */
template <JitBackend Backend>
class PagedHolder : public dr::TraversableBase {
public:
    using Float   = dr::JitArray<Backend, float>;
    using Pointer = typename dr::PagedArrayView<Float>::Pointer;

    PagedHolder() = default;
    PagedHolder(const std::vector<Float> &pages, size_t logical_size,
                size_t page_size)
        : m_view(pages, logical_size, page_size) { }

    dr::PagedArrayView<Float> &view() { return m_view; }
    const dr::PagedArrayView<Float> &view() const { return m_view; }

    void traverse_1_cb_ro(void *payload,
                          dr::detail::traverse_callback_ro fn) const override {
        for (const Float &page : m_view.pages())
            dr::traverse_1_fn_ro(page, payload, fn);
        dr::traverse_1_fn_ro(m_view.page_table(), payload, fn);
    }

    void traverse_1_cb_rw(void *payload,
                          dr::detail::traverse_callback_rw fn) override {
        using Access = dr::detail::paged_array_view_access<Float>;
        for (Float &page : Access::pages(m_view))
            dr::traverse_1_fn_rw(page, payload, fn);
        dr::traverse_1_fn_rw(Access::page_table(m_view), payload, fn);
    }

private:
    dr::PagedArrayView<Float> m_view;
};

/// Traversable wrapper around DiffPagedArrayView (see PagedHolder)
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

    void traverse_1_cb_ro(void *payload,
                          dr::detail::traverse_callback_ro fn) const override {
        const auto &primal = m_view.primal();
        for (const Float &page : primal.pages())
            dr::traverse_1_fn_ro(page, payload, fn);
        dr::traverse_1_fn_ro(primal.page_table(), payload, fn);
        dr::traverse_1_fn_ro(m_view.logical_proxy(), payload, fn);
    }

    void traverse_1_cb_rw(void *payload,
                          dr::detail::traverse_callback_rw fn) override {
        using Access = dr::detail::paged_array_view_access<Float>;
        auto &primal = m_view.primal();
        for (Float &page : Access::pages(primal))
            dr::traverse_1_fn_rw(page, payload, fn);
        dr::traverse_1_fn_rw(Access::page_table(primal), payload, fn);
        dr::traverse_1_fn_rw(m_view.logical_proxy(), payload, fn);
    }

private:
    View m_view;
};

template <JitBackend Backend> void bind(nb::module_ m) {
    using Float      = dr::JitArray<Backend, float>;
    using UInt32     = dr::JitArray<Backend, uint32_t>;
    using Mask       = dr::mask_t<Float>;
    using Float3     = dr::Array<Float, 3>;
    using View       = dr::PagedArrayView<Float>;
    using Holder     = PagedHolder<Backend>;
    using DiffFloat  = dr::DiffArray<Backend, float>;
    using DiffView   = dr::DiffPagedArrayView<DiffFloat>;
    using DiffHolder = DiffPagedHolder<Backend>;

    nb::class_<View>(m, "PagedArrayViewF32")
        .def(nb::init<>())
        .def(nb::init<const std::vector<Float> &, size_t, size_t>(),
             "pages"_a, "logical_size"_a, "page_size"_a)
        .def("size", &View::size)
        .def("page_size", &View::page_size)
        .def("page_count", &View::page_count)
        .def("power_of_two_page_size", &View::power_of_two_page_size)
        .def("page_length", &View::page_length)
        .def("update_page", &View::update_page)
        .def("rebuild_page_table", &View::rebuild_page_table);

    m.def("gather", [](const View &view, const UInt32 &index, const Mask &active) {
        return dr::gather<Float>(view, index, active);
    }, "view"_a, "index"_a, "active"_a);

    m.def("gather", [](const View &view, const UInt32 &index) {
        return dr::gather<Float>(view, index);
    }, "view"_a, "index"_a);

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

    nb::class_<DiffView>(m, "DiffPagedArrayViewF32")
        .def(nb::init<>())
        .def(nb::init<const std::vector<Float> &, size_t, size_t>(),
             "pages"_a, "logical_size"_a, "page_size"_a)
        .def("size", &DiffView::size)
        .def("update_page", &DiffView::update_page)
        .def("enable_grad",
             [](DiffView &v) { dr::enable_grad(v.logical_proxy()); })
        .def("grad_enabled",
             [](DiffView &v) { return dr::grad_enabled(v.logical_proxy()); })
        .def("proxy", [](DiffView &v) { return v.logical_proxy(); })
        .def("grad", [](DiffView &v) {
            return dr::grad(v.logical_proxy());
        })
        .def("set_grad", [](DiffView &v, const Float &value) {
            dr::set_grad(v.logical_proxy(), value);
        })
        .def("clear_grad", [](DiffView &v) {
            dr::clear_grad(v.logical_proxy());
        });

    m.def("gather_diff", [](const DiffView &view, const UInt32 &index,
                            const Mask &active) {
        return dr::gather<DiffFloat>(view, index, active);
    }, "view"_a, "index"_a, "active"_a);

    m.def("gather_diff", [](const DiffView &view, const UInt32 &index) {
        return dr::gather<DiffFloat>(view, index);
    }, "view"_a, "index"_a);

    auto diff_holder = nb::class_<DiffHolder>(
        m, "DiffPagedHolderF32",
        nb::intrusive_ptr<DiffHolder>(
            [](DiffHolder *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def(nb::init<>())
        .def(nb::init<const std::vector<Float> &, size_t, size_t>(),
             "pages"_a, "logical_size"_a, "page_size"_a)
        .def("size", [](const DiffHolder &h) { return h.view().size(); })
        .def("update_page", [](DiffHolder &h, size_t i, const Float &page) {
            h.view().update_page(i, page);
        })
        .def("enable_grad", [](DiffHolder &h) {
            dr::enable_grad(h.view().logical_proxy());
        })
        .def("grad", [](DiffHolder &h) {
            return dr::grad(h.view().logical_proxy());
        })
        .def("set_grad", [](DiffHolder &h, const Float &value) {
            dr::set_grad(h.view().logical_proxy(), value);
        })
        .def("clear_grad", [](DiffHolder &h) {
            dr::clear_grad(h.view().logical_proxy());
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
