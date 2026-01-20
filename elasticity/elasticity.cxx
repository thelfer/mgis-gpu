/*!
 * \file   elasticity.cxx
 * \brief
 * \author Thomas Helfer
 * \date   20/01/2026
 */

#include <span>
#include <vector>
#include <cstddef>
#include <cstdlib>
#include "MGIS/Config.hxx"

#ifdef MGIS_USE_STL_PARALLEL_ALGORITHMS
#ifdef __cpp_lib_parallel_algorithm
#define MGIS_HAS_STL_PARALLEL_ALGORITHMS
#endif /* __cpp_lib_parallel_algorithm */
#endif /* MGIS_USE_STL_PARALLEL_ALGORITHMS */

namespace mgis::gpu {

  bool sequential_kernel(std::span<mgis::real>,
                         std::span<const real>,
                         const std::size_t);

#ifdef MGIS_HAS_STL_PARALLEL_ALGORITHMS
  bool stlpar_kernel(std::span<mgis::real>,
                     std::span<const real>,
                     const std::size_t);
#endif /* MGIS_HAS_STL_PARALLEL_ALGORITHMS*/

  using KernelType = bool (*)(std::span<mgis::real>,
                              std::span<const real>,
                              const std::size_t);

  bool execute(const KernelType kernel, const std::size_t n) {
    const auto eto_values = std::vector<real>(6 * n, real{});
    auto sig_values = std::vector<real>(6 * n, real{});
    return kernel(sig_values, eto_values, n);
  }  // end of execute

}  // namespace mgis::gpu

int main() {
  auto success = true;
  success =
      mgis::gpu::execute(mgis::gpu::sequential_kernel, 100'000'000) && success;
#ifdef MGIS_HAS_STL_PARALLEL_ALGORITHMS
  success =
      mgis::gpu::execute(mgis::gpu::stlpar_kernel, 100'000'000) && success;
#endif /* MGIS_HAS_STL_PARALLEL_ALGORITHMS */
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
