#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <drjit/python.h>
#include <drjit/paged.h>

namespace nb = nanobind;
namespace dr = drjit;

using namespace nb::literals;

template <JitBackend Backend> void bind(nb::module_ m) {
    using Float  = dr::JitArray<Backend, float>;
    using UInt32 = dr::JitArray<Backend, uint32_t>;
    using Mask   = dr::mask_t<Float>;
    using Float3 = dr::Array<Float, 3>;
    using View   = dr::PagedArrayView<Float>;

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

}

NB_MODULE(paged_ext, m) {
    nb::module_::import_("drjit");

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
