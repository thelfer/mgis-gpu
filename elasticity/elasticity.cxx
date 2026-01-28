/*!
 * \file   elasticity.cxx
 * \brief
 * \author Thomas Helfer
 * \date   20/01/2026
 */

#include <span>
#include <chrono>
#include <vector>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <sstream>
#include <locale>
#include "MGIS/Config.hxx"

#ifdef MGIS_USE_STL_PARALLEL_ALGORITHMS
#ifdef __cpp_lib_parallel_algorithm
#define MGIS_HAS_STL_PARALLEL_ALGORITHMS
#endif /* __cpp_lib_parallel_algorithm */
#endif /* MGIS_USE_STL_PARALLEL_ALGORITHMS */

namespace mgis::gpu {

  //! \brief timing policy: set to true to enable warmup + timing measurement
  constexpr bool is_timed = true;

  struct thousand_sep : std::numpunct<char> {
    char do_thousands_sep() const override { return '\''; }
    std::string do_grouping() const override { return "\3"; }
  };

  template <typename T>
  std::string format_number(T value) {
    std::ostringstream oss;
    oss.imbue(std::locale(oss.getloc(), new thousand_sep));
    oss << value;
    return oss.str();
  }

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

  template <bool IsTimed = false>
  bool execute(const KernelType kernel,
               const std::size_t n,
               std::string_view program,
               std::string_view kernel_name) {
    const auto eto_values = std::vector<real>(6 * n, real{});
    auto sig_values = std::vector<real>(6 * n, real{});
    if constexpr (IsTimed) {
      // warmup run with only 1 integration point
      // this may sometimes be overkill, but it is sometimes very useful
      // to bypass lazy-instantiation in some -dlto and -rdc cases
      kernel(sig_values, eto_values, 1);
      // timed run
      const auto start = std::chrono::steady_clock::now();
      const auto success = kernel(sig_values, eto_values, n);
      const auto end = std::chrono::steady_clock::now();
      const auto elapsed_ms =
          std::chrono::duration<double, std::milli>(end - start).count();
      std::cout << program << " " << kernel_name << " kernel for "
                << format_number(n) << " integration points: "
                << format_number(elapsed_ms) << " ms\n";
      return success;
    } else {
      return kernel(sig_values, eto_values, n);
    }
  }  // end of execute

}  // namespace mgis::gpu

int main() {
  auto success = true;
  constexpr std::size_t n = 10'000'000;
  success = mgis::gpu::execute<mgis::gpu::is_timed>(
                mgis::gpu::sequential_kernel, n, "elasticity", "sequential") &&
            success;
#ifdef MGIS_HAS_STL_PARALLEL_ALGORITHMS
  success = mgis::gpu::execute<mgis::gpu::is_timed>(
                mgis::gpu::stlpar_kernel, n, "elasticity", "stlpar") &&
            success;
#endif /* MGIS_HAS_STL_PARALLEL_ALGORITHMS */
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
