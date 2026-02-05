/*!
 * \file   kernels2.cu
 * \brief
 * \author Thomas Helfer
 * \date   28/01/2026
 */

#include "TFEL/Material/Lame.hxx"
#include "MGIS/Function/SharedSpace.hxx"
#include "MGIS/Function/BasicLinearSpace.hxx"
#include "MGIS/Function/BasicLinearQuadratureSpace.hxx"
#include "MGIS/Function/StridedCoalescedMemoryAccessFunctionViewBase.hxx"
#include "MGIS/Function/TFEL/Tensors.hxx"

namespace mgis::gpu{

  using Stensor = tfel::math::stensor<3u, real>;
  using Stensor4 = tfel::math::st2tost2<3u, real>;
  using namespace mgis::function;
  using KCompositeView =
      StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 6 * 6, true>;
  using CompositeView =
      StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 6, true>;
  using ImmutableCompositeView =
      StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 6, false>;

  template <typename KType, typename SigType, typename EtoType>
  __device__ void elasticity(KType&& K, SigType&& sig, EtoType&& eto) {
    constexpr Stensor id = Stensor::Id();
    constexpr Stensor4 IxI = Stensor4::IxI();
    constexpr Stensor4 Id4 = Stensor4::Id();
    constexpr real young = real{150e9};
    constexpr real nu = real{1} / 3;
    constexpr real lambda = tfel::material::computeLambda(young, nu);
    constexpr real mu = tfel::material::computeMu(young, nu);
    sig = lambda * tfel::math::trace(eto) * id + 2 * mu * eto;
    K = lambda * IxI + 2 * mu * Id4;
  }

  __global__ void elasticity_kernel(KCompositeView K_view,
                                    CompositeView sig_view,
                                    ImmutableCompositeView eto_view,
                                    const size_type n) {
    const size_type idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
      auto K = K_view.get<0, Stensor4>(idx);
      auto sig = sig_view.get<0, Stensor>(idx);
      const auto eto = eto_view.get<0, Stensor>(idx);
      elasticity(K, sig, eto);
    }
  }

  bool cuda_kernel(std::span<real> K_values,
                   std::span<real> sig_values,
                   std::span<const real> eto_values,
                   const std::size_t n) {
    BasicLinearSpace space{n};
    ImmutableCompositeView eto_view{space, eto_values};
    CompositeView sig_view{space, sig_values};
    KCompositeView K_view{space, K_values};

    constexpr int threads_per_block = 32;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;

    elasticity_kernel<<<blocks, threads_per_block>>>(K_view, sig_view, eto_view, n);
    cudaDeviceSynchronize();

    return true;
  }

} // end of mgis::gpu
