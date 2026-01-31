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
    constexpr auto id = Stensor::Id();
    constexpr auto young = real{150e9};
    constexpr auto nu = real{1} / 3;
    constexpr auto lambda = tfel::material::computeLambda(young, nu);
    constexpr auto mu = tfel::material::computeMu(young, nu);
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
    auto space = BasicLinearSpace{n};
    auto eto_view = ImmutableCompositeView{space, eto_values};
    auto sig_view = CompositeView{space, sig_values};

    constexpr int threads_per_block = 32;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    elasticity_kernel<<<blocks, threads_per_block>>>(sig_view, eto_view, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CUDA kernel: %.3f ms (%.2f M elements/s)\n",
           milliseconds, n / (milliseconds * 1000.0));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return true;
  }

} // end of mgis::gpu