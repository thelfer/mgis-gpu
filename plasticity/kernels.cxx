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

inline constexpr auto plasticity = [](auto &sig, auto &eel, auto &p,
                                      const auto &eto_bts,
                                      const auto &eto_ets) {
  using Stensor = tfel::math::stensor<3u, real>;
  using stress = real;
  constexpr auto id = Stensor::Id();
  constexpr auto young = real{150e9};
  constexpr auto nu = real{1} / 3;
  constexpr auto s0 = 200e6;
  constexpr auto H = 3e9;
  constexpr auto lambda = tfel::material::computeLambda(young, nu);
  constexpr auto mu = tfel::material::computeMu(young, nu);
  const auto deto = eto_ets - eto_bts;
  eel += deto;
  const auto se = 2 * mu * deviator(eel);
  const auto seq_e = sigmaeq(se);
  const auto b = seq_e - s0 - H * p > stress{0};
  if(b){
    const auto iseq_e = 1/seq_e;
    const auto n = eval(3 * se / (2 * seq_e));
    const auto cste = 1 / (H + 3 * mu);
    const auto dp = (seq_e - s0 - H * p) * cste;
    eel -= dp * n;
    p += dp;
  }
  sig = lambda * trace(eel) * Stensor::Id() + 2 * mu * eel;
};

bool sequential_kernel(std::span<real> sig_values, std::span<real> isvs_values,
                       std::span<const real> eto_bts_values,
                       std::span<const real> eto_ets_values,
                       const std::size_t n) {
  using namespace mgis::function;
  using Stensor = tfel::math::stensor<3u, real>;
  auto space = BasicLinearSpace{n};
  using CompositeView =
      StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 6>;
  using CompositeView2 =
      StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 7>;
  using ImmutableCompositeView =
      StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 6,
                                                       false>;
  auto eto_bts_view = ImmutableCompositeView{space, eto_bts_values};
  auto eto_ets_view = ImmutableCompositeView{space, eto_ets_values};
  auto sig_view = CompositeView{space, sig_values};
  auto isvs_view = CompositeView2{space, isvs_values};
  for (size_type idx = 0; idx != getSpaceSize(space); ++idx) {
    auto sig = sig_view.get<0, Stensor>(idx);
    auto eel = isvs_view.get<0, Stensor>(idx);
    auto p = isvs_view.get<6, real>(idx);
    const auto eto_bts = eto_bts_view.get<0, Stensor>(idx);
    const auto eto_ets = eto_ets_view.get<0, Stensor>(idx);
    plasticity(sig, eel, p, eto_bts, eto_ets);
  }
  return true;
} // end of sequential_kernel

#ifdef MGIS_HAS_STL_PARALLEL_ALGORITHMS
bool stlpar_kernel(std::span<real> sig_values, std::span<real> isvs_values,
                   std::span<const real> eto_bts_values,
                   std::span<const real> eto_ets_values, const std::size_t n) {
  using namespace mgis::function;
  using Stensor = tfel::math::stensor<3u, real>;
  auto space = BasicLinearSpace{n};
  using CompositeView =
      StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 6>;
  using CompositeView2 =
      StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 7>;
    using ImmutableCompositeView =
        StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 6,
                                                         false>;
    auto eto_bts_view = ImmutableCompositeView{space, eto_bts_values};
    auto eto_ets_view = ImmutableCompositeView{space, eto_ets_values};
    auto sig_view = CompositeView{space, sig_values};
    auto isvs_view = CompositeView2{space, isvs_values};
    const auto iranges =
        std::views::iota(size_type{}, n);
    std::for_each(std::execution::par, iranges.begin(), iranges.end(),
                  [&](const size_type idx) mutable {
                    auto sig = sig_view.get<0, Stensor>(idx);
                    auto eel = isvs_view.get<0, Stensor>(idx);
                    auto p = isvs_view.get<6, real>(idx);
                    const auto eto_bts = eto_bts_view.get<0, Stensor>(idx);
                    const auto eto_ets = eto_ets_view.get<0, Stensor>(idx);
                    plasticity(sig, eel, p, eto_bts, eto_ets);
                  });
    return true;
} // end of stlpar_kernel
#endif /* MGIS_HAS_STL_PARALLEL_ALGORITHMS */

}  // end of namespace mgis::gpu