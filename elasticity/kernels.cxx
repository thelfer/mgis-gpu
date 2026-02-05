/*!
 * \file   kernels.cxx
 * \brief
 * \author Thomas Helfer
 * \date   20/01/2026
 */

#include <version>

#ifdef MGIS_USE_STL_PARALLEL_ALGORITHMS
#ifdef __cpp_lib_parallel_algorithm
#define MGIS_HAS_STL_PARALLEL_ALGORITHMS
#endif
#endif /* MGIS_USE_STL_PARALLEL_ALGORITHMS */

#ifdef MGIS_HAS_STL_PARALLEL_ALGORITHMS
#ifdef _NVHPC_STDPAR_GPU
#include <ranges>
#include <execution>
#else
#include <tbb/parallel_for.h>
#endif
#endif /* MGIS_HAS_STL_PARALLEL_ALGORITHMS */

#include "TFEL/Material/Lame.hxx"
#include "MGIS/Function/SharedSpace.hxx"
#include "MGIS/Function/BasicLinearSpace.hxx"
#include "MGIS/Function/BasicLinearQuadratureSpace.hxx"
#include "MGIS/Function/StridedCoalescedMemoryAccessFunctionViewBase.hxx"
#include "MGIS/Function/TFEL/Tensors.hxx"

namespace mgis::gpu {

  inline constexpr auto elasticity = [](auto& sig, const auto& eto) __attribute__((always_inline)) {
    using Stensor = tfel::math::stensor<3u, real>;
    constexpr auto id = Stensor::Id();
    constexpr auto young = real{150e9};
    constexpr auto nu = real{1} / 3;
    constexpr auto lambda = tfel::material::computeLambda(young, nu);
    constexpr auto mu = tfel::material::computeMu(young, nu);
    sig = lambda * tfel::math::trace(eto) * id + 2 * mu * eto;
  };

  bool
  sequential_kernel(std::span<real> sig_values,
                    std::span<const real> eto_values,
                    const std::size_t n) {
    using namespace mgis::function;
    using Stensor = tfel::math::stensor<3u, real>;
    auto space = BasicLinearSpace{n};
    using CompositeView =
        StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 6>;
    using ImmutableCompositeView =
        StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 6,
                                                         false>;
    auto eto_view = ImmutableCompositeView{space, eto_values};
    auto sig_view = CompositeView{space, sig_values};
    for (size_type idx = 0; idx != getSpaceSize(space); ++idx) {
      auto sig = sig_view.get<0, Stensor>(idx);
      const auto eto = eto_view.get<0, Stensor>(idx);
      elasticity(sig, eto);
    }
    return true;
  }  // end of sequential_kernel

#ifdef MGIS_HAS_STL_PARALLEL_ALGORITHMS
  bool stlpar_kernel(std::span<real> sig_values,
                     std::span<const real> eto_values,
                     const std::size_t n) {
    using namespace mgis::function;
    using Stensor = tfel::math::stensor<3u, real>;
    auto space = BasicLinearSpace{n};
    using CompositeView =
        StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 6>;
    using ImmutableCompositeView =
        StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 6,
                                                         false>;
    auto eto_view = ImmutableCompositeView{space, eto_values};
    auto sig_view = CompositeView{space, sig_values};
#ifdef _NVHPC_STDPAR_GPU
    const auto range = std::views::iota(size_type{0}, n);
    std::for_each(std::execution::par, range.begin(), range.end(),
                  [eto_view, sig_view](const size_type idx) mutable {
                    auto sig = sig_view.get<0, Stensor>(idx);
                    const auto eto = eto_view.get<0, Stensor>(idx);
                    elasticity(sig, eto);
                  });
#else
    tbb::parallel_for(size_type{0}, n, [&](size_type idx) {
      auto sig = sig_view.get<0, Stensor>(idx);
      const auto eto = eto_view.get<0, Stensor>(idx);
      elasticity(sig, eto);
    });
#endif
    return true;
  }  // end of stlpar_kernel
#endif /* MGIS_HAS_STL_PARALLEL_ALGORITHMS */

}  // end of namespace mgis::gpu
