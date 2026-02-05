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
#endif /* __cpp_lib_parallel_algorithm */
#endif /* MGIS_USE_STL_PARALLEL_ALGORITHMS */

#ifdef MGIS_HAS_STL_PARALLEL_ALGORITHMS
#ifdef _NVHPC_STDPAR_GPU
#include <ranges>
#include <execution>
#else
#include <tbb/parallel_for.h>
#endif
#endif /* MGIS_HAS_STL_PARALLEL_ALGORITHMS */

#include "TFEL/Math/T2toT2/ConvertToPK1Derivative.hxx"
#include "TFEL/Material/Lame.hxx"
#include "MGIS/Function/BasicLinearQuadratureSpace.hxx"
#include "MGIS/Function/BasicLinearSpace.hxx"
#include "MGIS/Function/SharedSpace.hxx"
#include "MGIS/Function/StridedCoalescedMemoryAccessFunctionViewBase.hxx"
#include "MGIS/Function/TFEL/Tensors.hxx"

namespace mgis::gpu {

inline constexpr auto signorini = [](auto &Dt, auto &sig, const auto &F1) __attribute__((always_inline)) {
  using namespace tfel::math;
  using namespace tfel::material;
  using real = double;
  using Tensor = tensor<3u, real>;
  using Stensor = stensor<3u, real>;
  using StressStensor = stensor<3u, real>;
  using Stensor4 = st2tost2<3u, real>;
  constexpr auto id = Stensor::Id();
  constexpr auto Id4 = Stensor4::Id();
  constexpr auto C10 = real{2.668};
  constexpr auto C01 = real{0.271};
  constexpr auto C20 = real{0.466};
  constexpr auto nu = real{0.499};
  constexpr auto K = 4 * (1 + nu) * (C10 + C01) / 3 / (1 - (2 * nu));
  //
  const auto J = det(F1);
  const auto C = computeRightCauchyGreenTensor(F1);
  const auto C2 = square(C);
  /* invariants and derivatives */
  const auto I1 = trace(C);
  const auto I2 = (I1 * I1 - trace(C2)) / 2;
  const auto dI2_dC = I1 * id - C;
  const auto I3 = J * J;
  const auto dI3_dC = C2 - I1 * C + I2 * id;
  /* volumetric part */
  // Pv = K*(J-1)*(J-1)/2
  const auto dPv_dJ = K * (J - 1);
  const StressStensor Sv = dPv_dJ / J * dI3_dC;
  /* isochoric part */
  // I1b = J^{-2/3}*I1 = I1/(sqrt[3]{I3})     = I1*iJb
  // I2b = J^{-4/3}*I2 = I2/(sqrt[3]{I3})^{2} = I2*iJb*iJb
  const auto iJb = 1 / cbrt(I3);
  const auto iJb2 = power<2>(iJb);
  const auto iJb4 = iJb2 * iJb2;
  const auto diJb_dI3 = -iJb4 / 3;
  const auto diJb_dC = diJb_dI3 * dI3_dC;
  const auto I1b = I1 * iJb;
  // const auto I2b = I2 * iJb * iJb;  // unused: dPi/dI2b = C01 is constant
  const auto dI1b_dC = iJb * id + I1 * diJb_dC;
  const auto dI2b_dC = iJb2 * dI2_dC + 2 * I2 * iJb * diJb_dC;
  const auto dPi_dI1b = C10 + 2 * C20 * (I1b - 3);
  const auto dPi_dI2b = C01;
  const StressStensor Si = 2 * (dPi_dI1b * dI1b_dC + dPi_dI2b * dI2b_dC);
  sig = convertSecondPiolaKirchhoffStressToCauchyStress(Sv + Si, F1);
  /* invariants second derivatives */
  const auto d2I3_dC2 = computeDeterminantSecondDerivative(C);
  const auto d2I2_dC2 = (id ^ id) - Id4;
  const auto iJb7 = iJb4 * power<3>(iJb);
  const auto d2iJb_dI32 = 4 * iJb7 / 9;
  const auto d2iJb_dC2 = d2iJb_dI32 * (dI3_dC ^ dI3_dC) + diJb_dI3 * d2I3_dC2;
  const auto d2I1b_dC2 = (id ^ diJb_dC) + (diJb_dC ^ id) + I1 * d2iJb_dC2;
  const auto d2I2b_dC2 = 2 * iJb * (dI2_dC ^ diJb_dC) + iJb2 * d2I2_dC2 +
                         2 * iJb * (diJb_dC ^ dI2_dC) +
                         2 * I2 * (diJb_dC ^ diJb_dC) +
                         2 * I2 * iJb * d2iJb_dC2;
  /* volumetric part */
  const auto d2Pv_dJ2 = K;
  const auto dSv_dC = (d2Pv_dJ2 - dPv_dJ / J) / (2 * I3) * (dI3_dC ^ dI3_dC) +
                      dPv_dJ / J * d2I3_dC2;
  /* isochoric part */
  const auto d2Pi_dI1b2 = 2 * C20;
  const auto dSi_dC = 2 * (d2Pi_dI1b2 * (dI1b_dC ^ dI1b_dC) +
                           dPi_dI1b * d2I1b_dC2 + dPi_dI2b * d2I2b_dC2);
  const auto dS_degl = eval((dSv_dC + dSi_dC) / 2);
  // This is probably totally inefficient
  Dt =
      convertSecondPiolaKirchhoffStressDerivativeToFirstPiolaKirchoffStressDerivative(
          dS_degl, Tensor{F1}, Stensor{sig});
};

bool sequential_kernel(std::span<real> Dt_values,
                       std::span<real> sig_values,
                       std::span<const real> F_values, const std::size_t n) {
  using namespace mgis::function;
  using Tensor = tfel::math::tensor<3u, real>;
  using Stensor = tfel::math::stensor<3u, real>;
  auto space = BasicLinearSpace{n};
  using CompositeView =
      StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 6>;
  using CompositeView2 =
      StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 81>;
  using ImmutableCompositeView =
      StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 9,
                                                       false>;
  auto F_view = ImmutableCompositeView{space, F_values};
  auto Dt_view = CompositeView2{space, Dt_values};
  auto sig_view = CompositeView{space, sig_values};
  for (size_type idx = 0; idx != getSpaceSize(space); ++idx) {
    auto sig = sig_view.get<0, Stensor>(idx);
    auto Dt = Dt_view.get<0, tfel::math::t2tot2<3u, real>>(idx);
    const auto F = F_view.get<0, Tensor>(idx);
    signorini(Dt, sig, F);
  }
  return true;
} // end of sequential_kernel

#ifdef MGIS_HAS_STL_PARALLEL_ALGORITHMS
bool stlpar_kernel(std::span<real> Dt_values,
                   std::span<real> sig_values,
                   std::span<const real> F_values,
                   const std::size_t n) {
  using namespace mgis::function;
  using Tensor = tfel::math::tensor<3u, real>;
  using Stensor = tfel::math::stensor<3u, real>;
  auto space = BasicLinearSpace{n};
  using CompositeView =
      StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 6>;
  using CompositeView2 =
      StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 81>;
  using ImmutableCompositeView =
      StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 9,
                                                       false>;
  auto F_view = ImmutableCompositeView{space, F_values};
  auto Dt_view = CompositeView2{space, Dt_values};
  auto sig_view = CompositeView{space, sig_values};
#ifdef _NVHPC_STDPAR_GPU
  const auto iranges = std::views::iota(size_type{}, n);
  std::for_each(std::execution::par, iranges.begin(), iranges.end(),
                [F_view, Dt_view, sig_view](const size_type idx) mutable {
                  auto Dt = Dt_view.get<0, tfel::math::t2tot2<3u, real>>(idx);
                  auto sig = sig_view.get<0, Stensor>(idx);
                  const auto F = F_view.get<0, Tensor>(idx);
                  signorini(Dt, sig, F);
                });
#else
  tbb::parallel_for(size_type{0}, n, [&](size_type idx) {
    auto Dt = Dt_view.get<0, tfel::math::t2tot2<3u, real>>(idx);
    auto sig = sig_view.get<0, Stensor>(idx);
    const auto F = F_view.get<0, Tensor>(idx);
    signorini(Dt, sig, F);
  });
#endif
  return true;
} // end of stlpar_kernel
#endif /* MGIS_HAS_STL_PARALLEL_ALGORITHMS */

} // end of namespace mgis::gpu