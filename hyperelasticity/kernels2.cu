/*!
 * \file   kernels2.cu
 * \brief
 * \author Tristan Chenaille
 * \date   01/02/2026
 */

#include "TFEL/Math/T2toT2/ConvertToPK1Derivative.hxx"
#include "TFEL/Material/Lame.hxx"
#include "MGIS/Function/BasicLinearQuadratureSpace.hxx"
#include "MGIS/Function/BasicLinearSpace.hxx"
#include "MGIS/Function/SharedSpace.hxx"
#include "MGIS/Function/StridedCoalescedMemoryAccessFunctionViewBase.hxx"
#include "MGIS/Function/TFEL/Tensors.hxx"

namespace mgis::gpu {

  using Tensor = tfel::math::tensor<3u, real>;
  using Stensor = tfel::math::stensor<3u, real>;
  using StressStensor = tfel::math::stensor<3u, real>;
  using Stensor4 = tfel::math::st2tost2<3u, real>;
  using Tensor4 = tfel::math::t2tot2<3u, real>;
  using namespace mgis::function;
  using CompositeView =
      StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 6, true>;
  using CompositeView2 =
      StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 81, true>;
  using ImmutableCompositeView =
      StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 9, false>;

  template <typename DtType, typename SigType, typename FType>
  __device__ void signorini(DtType&& Dt, SigType&& sig, FType&& F1) {
    constexpr Stensor id = Stensor::Id();
    constexpr Stensor4 Id4 = Stensor4::Id();
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
    /* invariants second derivatives */
    const Stensor4 d2I3_dC2 = tfel::math::computeDeterminantSecondDerivative(C);
    const Stensor4 d2I2_dC2 = (id ^ id) - Id4;
    const real iJb7 = iJb4 * tfel::math::power<3>(iJb);
    const real d2iJb_dI32 = 4 * iJb7 / 9;
    const Stensor4 d2iJb_dC2 = d2iJb_dI32 * (dI3_dC ^ dI3_dC) + diJb_dI3 * d2I3_dC2;
    const Stensor4 d2I1b_dC2 = (id ^ diJb_dC) + (diJb_dC ^ id) + I1 * d2iJb_dC2;
    const Stensor4 d2I2b_dC2 = 2 * iJb * (dI2_dC ^ diJb_dC) + iJb2 * d2I2_dC2 +
                               2 * iJb * (diJb_dC ^ dI2_dC) +
                               2 * I2 * (diJb_dC ^ diJb_dC) +
                               2 * I2 * iJb * d2iJb_dC2;
    /* volumetric part */
    const real d2Pv_dJ2 = K;
    const Stensor4 dSv_dC = (d2Pv_dJ2 - dPv_dJ / J) / (2 * I3) * (dI3_dC ^ dI3_dC) +
                            dPv_dJ / J * d2I3_dC2;
    /* isochoric part */
    const real d2Pi_dI1b2 = 2 * C20;
    const Stensor4 dSi_dC = 2 * (d2Pi_dI1b2 * (dI1b_dC ^ dI1b_dC) +
                                 dPi_dI1b * d2I1b_dC2 + dPi_dI2b * d2I2b_dC2);
    const Stensor4 dS_degl = tfel::math::eval((dSv_dC + dSi_dC) / 2);
    // This is probably totally inefficient
    Dt =
        tfel::math::convertSecondPiolaKirchhoffStressDerivativeToFirstPiolaKirchoffStressDerivative(
            dS_degl, Tensor{F1}, Stensor{sig});
  }

  __global__ void signorini_kernel(CompositeView2 Dt_view,
                                   CompositeView sig_view,
                                   ImmutableCompositeView F_view,
                                   const size_type n) {
    const size_type idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
      auto Dt = Dt_view.get<0, Tensor4>(idx);
      auto sig = sig_view.get<0, Stensor>(idx);
      const auto F = F_view.get<0, Tensor>(idx);
      signorini(Dt, sig, F);
    }
  }

  bool cuda_kernel(std::span<real> Dt_values,
                   std::span<real> sig_values,
                   std::span<const real> F_values,
                   const std::size_t n) {
    BasicLinearSpace space{n};
    ImmutableCompositeView F_view{space, F_values};
    CompositeView2 Dt_view{space, Dt_values};
    CompositeView sig_view{space, sig_values};

    constexpr int threads_per_block = 32;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;

    signorini_kernel<<<blocks, threads_per_block>>>(Dt_view, sig_view, F_view, n);
    cudaDeviceSynchronize();

    return true;
  }

} // end of mgis::gpu
