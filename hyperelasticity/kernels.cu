/*!
 * \file   kernels.cu
 * \brief
 * \author Tristan Chenaille
 * \date   01/02/2026
 */

#include "MGIS/Function/BasicLinearQuadratureSpace.hxx"
#include "MGIS/Function/BasicLinearSpace.hxx"
#include "MGIS/Function/SharedSpace.hxx"
#include "MGIS/Function/StridedCoalescedMemoryAccessFunctionViewBase.hxx"
#include "MGIS/Function/TFEL/Tensors.hxx"
#include "TFEL/Material/Lame.hxx"

namespace mgis::gpu {

  using Tensor = tfel::math::tensor<3u, real>;
  using Stensor = tfel::math::stensor<3u, real>;
  using StressStensor = tfel::math::stensor<3u, real>;
  using namespace mgis::function;
  using CompositeView =
      StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 6, true>;
  using ImmutableCompositeView =
      StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 9, false>;

  template <typename SigType, typename FType>
  __device__ void signorini(SigType&& sig, FType&& F1) {
    constexpr Stensor id = Stensor::Id();
    constexpr real C10 = real{2.668};
    constexpr real C01 = real{0.271};
    constexpr real C20 = real{0.466};
    constexpr real nu = real{0.499};
    constexpr real K = 4 * (1 + nu) * (C10 + C01) / 3 / (1 - (2 * nu));
    //
    const real J = tfel::math::det(F1);
    const Stensor C = tfel::math::computeRightCauchyGreenTensor(F1);
    const Stensor C2 = tfel::math::square(C);
    /* invariants and derivatives */
    const real I1 = tfel::math::trace(C);
    const real I2 = (I1 * I1 - tfel::math::trace(C2)) / 2;
    const Stensor dI2_dC = I1 * id - C;
    const real I3 = J * J;
    const Stensor dI3_dC = C2 - I1 * C + I2 * id;
    /* volumetric part */
    // Pv = K*(J-1)*(J-1)/2
    const real dPv_dJ = K * (J - 1);
    const StressStensor Sv = dPv_dJ / J * dI3_dC;
    /* isochoric part */
    // I1b = J^{-2/3}*I1 = I1/(sqrt[3]{I3})     = I1*iJb
    // I2b = J^{-4/3}*I2 = I2/(sqrt[3]{I3})^{2} = I2*iJb*iJb
    const real iJb = 1 / cbrt(I3);
    const real iJb2 = tfel::math::power<2>(iJb);
    const real iJb4 = iJb2 * iJb2;
    const real diJb_dI3 = -iJb4 / 3;
    const Stensor diJb_dC = diJb_dI3 * dI3_dC;
    const real I1b = I1 * iJb;
    // const real I2b = I2 * iJb * iJb;  // unused: dPi/dI2b = C01 is constant
    const Stensor dI1b_dC = iJb * id + I1 * diJb_dC;
    const Stensor dI2b_dC = iJb2 * dI2_dC + 2 * I2 * iJb * diJb_dC;
    const real dPi_dI1b = C10 + 2 * C20 * (I1b - 3);
    const real dPi_dI2b = C01;
    const StressStensor Si = 2 * (dPi_dI1b * dI1b_dC + dPi_dI2b * dI2b_dC);
    sig = tfel::math::convertSecondPiolaKirchhoffStressToCauchyStress(Sv + Si, F1);
  }

  __global__ void signorini_kernel(CompositeView sig_view,
                                   ImmutableCompositeView F_view,
                                   const size_type n) {
    const size_type idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
      auto sig = sig_view.get<0, Stensor>(idx);
      const auto F = F_view.get<0, Tensor>(idx);
      signorini(sig, F);
    }
  }

  bool cuda_kernel(std::span<real> sig_values,
                   std::span<const real> F_values,
                   const std::size_t n) {
    BasicLinearSpace space{n};
    ImmutableCompositeView F_view{space, F_values};
    CompositeView sig_view{space, sig_values};

    constexpr int threads_per_block = 32;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;

    signorini_kernel<<<blocks, threads_per_block>>>(sig_view, F_view, n);
    cudaDeviceSynchronize();

    return true;
  }

} // end of mgis::gpu
