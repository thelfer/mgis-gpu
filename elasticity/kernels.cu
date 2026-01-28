/*!
 * \file   kernels.cu
 * \brief    
 * \author Thomas Helfer
 * \date   28/01/2026
 */

#include "TFEL/Material/Lame.hxx"
#include "MGIS/Function/SharedSpace.hxx"
#include "MGIS/Function/BasicLinearSpace.hxx"
#include "MGIS/Function/BasicLinearQuadratureSpace.hxx"
#include "MGIS/Function/StridedCoalescedMemoryAccessFunctionViewBase.hxx"
#include "MGIS/Function/Tensors.hxx"

namespace mgis::gpu{

  inline constexpr auto elasticity = [](auto& sig, const auto& eto) {
    using Stensor = tfel::math::stensor<3u, real>;
    constexpr auto id = Stensor::Id();
    constexpr auto young = real{150e9};
    constexpr auto nu = real{1} / 3;
    constexpr auto lambda = tfel::material::computeLambda(young, nu);
    constexpr auto mu = tfel::material::computeMu(young, nu);
    sig = lambda * tfel::math::trace(eto) * id + 2 * mu * eto;
  };

  bool cuda_kernel(std::span<real> sig_values, std::span<const real> eto_values,
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
    // write loop
    return true;
  } // end of sequential_kernel

} // end of mgis::gpu