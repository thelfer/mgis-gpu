/*!
 * \file   kernels2.cxx
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
#include "MGIS/Function/Tensors.hxx"

namespace mgis::gpu {

  inline constexpr auto plasticity = [](auto &Dt, auto &sig, auto &eel, auto &p,
                                        const auto &eto_bts,
                                        const auto &eto_ets) __attribute__((always_inline)) {
    using namespace tfel::math;
    using namespace tfel::material;
    using Stensor = stensor<3u, real>;
    using Stensor4 = st2tost2<3u, real>;
    using stress = real;
    constexpr auto id = Stensor::Id();
    constexpr auto IxI = Stensor4::IxI();
    constexpr auto Id4 = Stensor4::Id();
    constexpr auto M = Stensor4::M();
    constexpr auto young = real{150e9};
    constexpr auto nu = real{1} / 3;
    constexpr auto s0 = stress{200e6};
    constexpr auto H = stress{3e9};
    constexpr auto lambda = computeLambda(young, nu);
    constexpr auto mu = computeMu(young, nu);
    const auto deto = eto_ets - eto_bts;
    eel += deto;
    const auto se = 2 * mu * deviator(eel);
    const auto seq_e = sigmaeq(se);
    const auto b = seq_e - s0 - H * p > stress{0};
    if (b) {
      const auto iseq_e = 1 / seq_e;
      const auto n = eval(3 * se / (2 * seq_e));
      const auto cste = 1 / (H + 3 * mu);
      const auto dp = (seq_e - s0 - H * p) * cste;
      eel -= dp * n;
      p += dp;
      Dt = (lambda * IxI + 2 * mu * Id4 -
            4 * mu * mu *
                (dp * iseq_e * (M - (n ^ n)) + cste * (n ^ n)));
    } else {
      Dt = lambda * IxI + 2 * mu * Id4;
    }
    sig = lambda * trace(eel) * id + 2 * mu * eel;
  };

  bool sequential_kernel(std::span<real> Dt_values,
                         std::span<real> sig_values,
                         std::span<real> isvs_values,
                         std::span<const real> eto_bts_values,
                         std::span<const real> eto_ets_values,
                         const std::size_t n) {
    using namespace mgis::function;
    using Stensor = tfel::math::stensor<3u, real>;
    using Stensor4 = tfel::math::st2tost2<3u, real>;
    auto space = BasicLinearSpace{n};
    using CompositeView =
        StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 6>;
    using CompositeView2 =
        StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 7>;
    using CompositeView3 =
        StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 36>;
    using ImmutableCompositeView =
        StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 6,
                                                         false>;
    auto eto_bts_view = ImmutableCompositeView{space, eto_bts_values};
    auto eto_ets_view = ImmutableCompositeView{space, eto_ets_values};
    auto Dt_view = CompositeView3{space, Dt_values};
    auto sig_view = CompositeView{space, sig_values};
    auto isvs_view = CompositeView2{space, isvs_values};
    for (size_type idx = 0; idx != getSpaceSize(space); ++idx) {
      auto Dt = Dt_view.get<0, Stensor4>(idx);
      auto sig = sig_view.get<0, Stensor>(idx);
      auto eel = isvs_view.get<0, Stensor>(idx);
      auto p = isvs_view.get<6, real>(idx);
      const auto eto_bts = eto_bts_view.get<0, Stensor>(idx);
      const auto eto_ets = eto_ets_view.get<0, Stensor>(idx);
      plasticity(Dt, sig, eel, p, eto_bts, eto_ets);
    }
    return true;
  }  // end of sequential_kernel

#ifdef MGIS_HAS_STL_PARALLEL_ALGORITHMS
  bool stlpar_kernel(std::span<real> Dt_values,
                     std::span<real> sig_values,
                     std::span<real> isvs_values,
                     std::span<const real> eto_bts_values,
                     std::span<const real> eto_ets_values,
                     const std::size_t n) {
    using namespace mgis::function;
    using Stensor = tfel::math::stensor<3u, real>;
    using Stensor4 = tfel::math::st2tost2<3u, real>;
    auto space = BasicLinearSpace{n};
    using CompositeView =
        StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 6>;
    using CompositeView2 =
        StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 7>;
    using CompositeView3 =
        StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 36>;
    using ImmutableCompositeView =
        StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 6,
                                                         false>;
    auto eto_bts_view = ImmutableCompositeView{space, eto_bts_values};
    auto eto_ets_view = ImmutableCompositeView{space, eto_ets_values};
    auto Dt_view = CompositeView3{space, Dt_values};
    auto sig_view = CompositeView{space, sig_values};
    auto isvs_view = CompositeView2{space, isvs_values};
#ifdef _NVHPC_STDPAR_GPU
    const auto range = std::views::iota(size_type{0}, n);
    std::for_each(std::execution::par, range.begin(), range.end(),
                  [eto_bts_view, eto_ets_view, Dt_view, sig_view, isvs_view](const size_type idx) mutable {
                    auto Dt = Dt_view.get<0, Stensor4>(idx);
                    auto sig = sig_view.get<0, Stensor>(idx);
                    auto eel = isvs_view.get<0, Stensor>(idx);
                    auto p = isvs_view.get<6, real>(idx);
                    const auto eto_bts = eto_bts_view.get<0, Stensor>(idx);
                    const auto eto_ets = eto_ets_view.get<0, Stensor>(idx);
                    plasticity(Dt, sig, eel, p, eto_bts, eto_ets);
                  });
#else
    tbb::parallel_for(size_type{0}, n, [&](size_type idx) {
      auto Dt = Dt_view.get<0, Stensor4>(idx);
      auto sig = sig_view.get<0, Stensor>(idx);
      auto eel = isvs_view.get<0, Stensor>(idx);
      auto p = isvs_view.get<6, real>(idx);
      const auto eto_bts = eto_bts_view.get<0, Stensor>(idx);
      const auto eto_ets = eto_ets_view.get<0, Stensor>(idx);
      plasticity(Dt, sig, eel, p, eto_bts, eto_ets);
    });
#endif
    return true;
  }  // end of stlpar_kernel
#endif /* MGIS_HAS_STL_PARALLEL_ALGORITHMS */

}  // end of namespace mgis::gpu
