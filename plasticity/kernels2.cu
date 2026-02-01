/*!
 * \file   kernels2.cu
 * \brief
 * \author Tristan Chenaille
 * \date   01/02/2026
 */

#include "TFEL/Material/Lame.hxx"
#include "MGIS/Function/SharedSpace.hxx"
#include "MGIS/Function/BasicLinearSpace.hxx"
#include "MGIS/Function/BasicLinearQuadratureSpace.hxx"
#include "MGIS/Function/StridedCoalescedMemoryAccessFunctionViewBase.hxx"
#include "MGIS/Function/Tensors.hxx"

namespace mgis::gpu {

  using Stensor = tfel::math::stensor<3u, real>;
  using Stensor4 = tfel::math::st2tost2<3u, real>;
  using stress = real;
  using namespace mgis::function;
  using CompositeView =
      StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 6, true>;
  using CompositeView2 =
      StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 7, true>;
  using CompositeView3 =
      StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 36, true>;
  using ImmutableCompositeView =
      StridedCoalescedMemoryAccessCompositeTensorsView<BasicLinearSpace, 6, false>;

  template <typename DtType, typename SigType, typename EelType, typename PType,
            typename EtoBtsType, typename EtoEtsType>
  __device__ void plasticity(DtType&& Dt, SigType&& sig, EelType&& eel, PType&& p,
                             EtoBtsType&& eto_bts, EtoEtsType&& eto_ets) {
    constexpr auto id = Stensor::Id();
    constexpr auto IxI = Stensor4::IxI();
    constexpr auto Id4 = Stensor4::Id();
    constexpr auto M = Stensor4::M();
    constexpr auto young = real{150e9};
    constexpr auto nu = real{1} / 3;
    constexpr auto s0 = stress{200e6};
    constexpr auto H = stress{3e9};
    constexpr auto lambda = tfel::material::computeLambda(young, nu);
    constexpr auto mu = tfel::material::computeMu(young, nu);
    const auto deto = eto_ets - eto_bts;
    eel += deto;
    const auto se = 2 * mu * tfel::math::deviator(eel);
    const auto seq_e = tfel::math::sigmaeq(se);
    const auto b = seq_e - s0 - H * p > stress{0};
    if (b) {
      const auto iseq_e = 1 / seq_e;
      const auto n = tfel::math::eval(3 * se / (2 * seq_e));
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
    sig = lambda * tfel::math::trace(eel) * id + 2 * mu * eel;
  }

  __global__ void plasticity_kernel(CompositeView3 Dt_view,
                                    CompositeView sig_view,
                                    CompositeView2 isvs_view,
                                    ImmutableCompositeView eto_bts_view,
                                    ImmutableCompositeView eto_ets_view,
                                    const size_type n) {
    const size_type idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
      auto Dt = Dt_view.get<0, Stensor4>(idx);
      auto sig = sig_view.get<0, Stensor>(idx);
      auto eel = isvs_view.get<0, Stensor>(idx);
      auto p = isvs_view.get<6, real>(idx);
      const auto eto_bts = eto_bts_view.get<0, Stensor>(idx);
      const auto eto_ets = eto_ets_view.get<0, Stensor>(idx);
      plasticity(Dt, sig, eel, p, eto_bts, eto_ets);
    }
  }

  bool cuda_kernel(std::span<real> Dt_values,
                   std::span<real> sig_values,
                   std::span<real> isvs_values,
                   std::span<const real> eto_bts_values,
                   std::span<const real> eto_ets_values,
                   const std::size_t n) {
    auto space = BasicLinearSpace{n};
    auto eto_bts_view = ImmutableCompositeView{space, eto_bts_values};
    auto eto_ets_view = ImmutableCompositeView{space, eto_ets_values};
    auto Dt_view = CompositeView3{space, Dt_values};
    auto sig_view = CompositeView{space, sig_values};
    auto isvs_view = CompositeView2{space, isvs_values};

    constexpr int threads_per_block = 32;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;

    plasticity_kernel<<<blocks, threads_per_block>>>(Dt_view, sig_view, isvs_view,
                                                     eto_bts_view, eto_ets_view, n);
    cudaDeviceSynchronize();

    return true;
  }

} // end of mgis::gpu
