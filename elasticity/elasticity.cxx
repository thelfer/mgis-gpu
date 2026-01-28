/*!
 * \file   elasticity.cxx
 * \brief
 * \author Thomas Helfer
 * \date   20/01/2026
 */

#ifdef MGIS_USE_STL_PARALLEL_ALGORITHMS
#ifdef __cpp_lib_parallel_algorithm
#define MGIS_HAS_STL_PARALLEL_ALGORITHMS
#endif /* __cpp_lib_parallel_algorithm */
#endif /* MGIS_USE_STL_PARALLEL_ALGORITHMS */

#include <span>
#include <vector>
#include <cstddef>
#include <cstdlib>
#include "MGIS/Config.hxx"

#ifdef MGIS_GPU_HAS_CUDA_SUPPORT
#include <cuda_runtime.h>
#endif /* MGIS_GPU_HAS_CUDA_SUPPORT */

namespace mgis::gpu {

  bool sequential_kernel(std::span<mgis::real>,
                         std::span<const real>,
                         const std::size_t);

#ifdef MGIS_HAS_STL_PARALLEL_ALGORITHMS
  bool stlpar_kernel(std::span<mgis::real>,
                     std::span<const real>,
                     const std::size_t);
#endif /* MGIS_HAS_STL_PARALLEL_ALGORITHMS*/

#ifdef MGIS_GPU_HAS_CUDA_SUPPORT
  bool cuda_kernel(std::span<mgis::real>, std::span<const real>,
                   const std::size_t);
#endif /* MGIS_GPU_HAS_CUDA_SUPPORT */

  using KernelType = bool (*)(std::span<mgis::real>,
                              std::span<const real>,
                              const std::size_t);

  bool execute(const KernelType kernel, const std::size_t n) {
    const auto eto_values = std::vector<real>(6 * n, real{});
    auto sig_values = std::vector<real>(6 * n, real{});
    return kernel(sig_values, eto_values, n);
  }  // end of execute

#ifdef MGIS_GPU_HAS_CUDA_SUPPORT
  std::span<real> allocate(const std::size_t n) {
    real *ptr;
    cudaMalloc(&ptr, n);
    return std::span<real>(ptr, n);
  }

  void deallocate(std::span<real> s) {
    if (!s.empty()) {
      cudaFree(s.data());
    }
  }

  bool cuda_execute(const KernelType kernel, const std::size_t n) {
    const auto eto_values = allocate(6 * n);
    auto sig_values = allocate(6 * n);
    auto success = kernel(sig_values, eto_values, n);
    deallocate(eto_values);
    deallocate(sig_values);
    return success;
  }    // end of execute
#endif /* MGIS_GPU_HAS_CUDA_SUPPORT */

}  // namespace mgis::gpu

int main() {
  auto success = true;
  success =
      mgis::gpu::execute(mgis::gpu::sequential_kernel, 100'000'000) && success;
#ifdef MGIS_HAS_STL_PARALLEL_ALGORITHMS
  success =
      mgis::gpu::execute(mgis::gpu::stlpar_kernel, 100'000'000) && success;
#endif /* MGIS_HAS_STL_PARALLEL_ALGORITHMS */
#ifdef MGIS_GPU_HAS_CUDA_SUPPORT
  success =
      mgis::gpu::cuda_execute(mgis::gpu::cuda_kernel, 100'000'000) && success;
#endif /* MGIS_GPU_HAS_CUDA_SUPPORT */
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
