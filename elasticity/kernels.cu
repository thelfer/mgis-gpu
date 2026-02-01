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

  using Stensor = tfel::math::stensor<3u, real>;
  using namespace mgis::function;
  using CompositeView =
      StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 6, true>;
  using ImmutableCompositeView =
      StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 6, false>;

  template <typename SigType, typename EtoType>
  __device__ void elasticity(SigType&& sig, EtoType&& eto) {
    constexpr Stensor id = Stensor::Id();
    constexpr real young = real{150e9};
    constexpr real nu = real{1} / 3;
    constexpr real lambda = tfel::material::computeLambda(young, nu);
    constexpr real mu = tfel::material::computeMu(young, nu);
    sig = lambda * tfel::math::trace(eto) * id + 2 * mu * eto;
  }

  __global__ void elasticity_kernel(CompositeView sig_view,
                                    ImmutableCompositeView eto_view,
                                    const size_type n) {
    const size_type idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
      auto sig = sig_view.get<0, Stensor>(idx);
      const auto eto = eto_view.get<0, Stensor>(idx);
      elasticity(sig, eto);
    }
  }

  bool cuda_kernel(std::span<real> sig_values,
                   std::span<const real> eto_values,
                   const std::size_t n) {
    BasicLinearSpace space{n};
    ImmutableCompositeView eto_view{space, eto_values};
    CompositeView sig_view{space, sig_values};

    constexpr int threads_per_block = 32;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;

    elasticity_kernel<<<blocks, threads_per_block>>>(sig_view, eto_view, n);
    cudaDeviceSynchronize();

    return true;
  }

} // end of mgis::gpu
