/*!
 * \file   kernels.cxx
 * \brief    
 * \author Thomas Helfer
 * \date   20/01/2026
 */

#ifdef MGIS_USE_STL_PARALLEL_ALGORITHMS
#ifdef __cpp_lib_parallel_algorithm
#define MGIS_HAS_STL_PARALLEL_ALGORITHMS
#endif /* __cpp_lib_parallel_algorithm */
#endif /* MGIS_USE_STL_PARALLEL_ALGORITHMS */

#ifdef MGIS_HAS_STL_PARALLEL_ALGORITHMS
#include <ranges>
#include <execution>
#endif /* MGIS_HAS_STL_PARALLEL_ALGORITHMS */

#include "TFEL/Material/Lame.hxx"
#include "MGIS/Function/SharedSpace.hxx"
#include "MGIS/Function/BasicLinearSpace.hxx"
#include "MGIS/Function/BasicLinearQuadratureSpace.hxx"
#include "MGIS/Function/StridedCoalescedMemoryAccessFunctionViewBase.hxx"
#include "MGIS/Function/Tensors.hxx"

namespace mgis::gpu{

  inline constexpr auto elasticity = [](auto& K, auto& sig, const auto& eto) {
    using namespace tfel::math;
    using namespace tfel::material;
    using Stensor = stensor<3u, real>;
    using Stensor4 = st2tost2<3u, real>;
    constexpr auto id = Stensor::Id();
    constexpr auto young = real{150e9};
    constexpr auto nu = real{1} / 3;
    constexpr auto lambda = computeLambda(young, nu);
    constexpr auto mu = computeMu(young, nu);
    sig = lambda * trace(eto) * id + 2 * mu * eto;
    K = lambda * Stensor4::IxI() + 2 * mu * Stensor4::Id();
  };

  bool
  sequential_kernel(std::span<real> K_values,
                    std::span<real> sig_values,
                    std::span<const real> eto_values,
                    const std::size_t n) {
    using namespace mgis::function;
    using Stensor = tfel::math::stensor<3u, real>;
    using Stensor4 = tfel::math::st2tost2<3u, real>;
    auto space = BasicLinearSpace{n};
    using KCompositeView =
        StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 6 * 6>;
    using CompositeView =
        StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 6>;
    using ImmutableCompositeView =
        StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 6,
                                                         false>;
    const auto eto_view = ImmutableCompositeView{space, eto_values};
    auto sig_view = CompositeView{space, sig_values};
    auto K_view = KCompositeView{space, K_values};
    for (size_type idx = 0; idx != getSpaceSize(space); ++idx) {
      auto K = K_view.get<0, Stensor4>(idx);
      auto sig = sig_view.get<0, Stensor>(idx);
      const auto eto = eto_view.get<0, Stensor>(idx);
      elasticity(K, sig, eto);
    }
    return true;
  }  // end of sequential_kernel

#ifdef MGIS_HAS_STL_PARALLEL_ALGORITHMS
  bool stlpar_kernel(std::span<real> K_values,
                     std::span<real> sig_values,
                         std::span<const real> eto_values,
                         const std::size_t n) {
    using namespace mgis::function;
    using Stensor = tfel::math::stensor<3u, real>;
    using Stensor4 = tfel::math::st2tost2<3u, real>;
    using KCompositeView =
        StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 6 * 6>;
    auto space = BasicLinearSpace{n};
    using CompositeView =
        StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 6>;
    using ImmutableCompositeView =
        StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 6,
                                                         false>;
    const auto eto_view = ImmutableCompositeView{space, eto_values};
    auto K_view = KCompositeView{space, K_values};
    auto sig_view = CompositeView{space, sig_values};
    const auto iranges =
        std::views::iota(size_type{}, n);
    std::for_each(std::execution::par, iranges.begin(), iranges.end(),
                  [eto_view, K_view, sig_view](const size_type idx) mutable {
                    auto K = K_view.get<0, Stensor4>(idx);
                    auto sig = sig_view.get<0, Stensor>(idx);
                    const auto eto = eto_view.get<0, Stensor>(idx);
                    elasticity(K, sig, eto);
                  });
    return true;
  }  // end of stlpar_kernel
#endif /* MGIS_HAS_STL_PARALLEL_ALGORITHMS */

}  // end of namespace mgis::gpu