#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <drjit/python.h>
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

template <JitBackend Backend> void bind(nb::module_ m) {
    using Float  = dr::JitArray<Backend, float>;
    using UInt32 = dr::JitArray<Backend, uint32_t>;
    using Mask   = dr::mask_t<Float>;
    using Float3 = dr::Array<Float, 3>;
    using View   = dr::PagedArrayView<Float>;
    using Holder = PagedHolder<Backend>;

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
