/*!
 * \file   signorini.cxx
 * \brief
 * \author Thomas Helfer
 * \date   28/01/2026
 */

#include <version>
#include <span>
#include <chrono>
#include <vector>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <sstream>
#include <locale>
#include <iomanip>
#include <type_traits>
#include "MGIS/Config.hxx"

#ifdef MGIS_USE_STL_PARALLEL_ALGORITHMS
#ifdef __cpp_lib_parallel_algorithm
#define MGIS_HAS_STL_PARALLEL_ALGORITHMS
#endif
#endif /* MGIS_USE_STL_PARALLEL_ALGORITHMS */

#if defined(_NVHPC_STDPAR_GPU) || defined(MGIS_GPU_HAS_CUDA_SUPPORT)
#include <cuda_runtime.h>
#endif

#if defined(MGIS_HAS_STL_PARALLEL_ALGORITHMS) && !defined(_NVHPC_STDPAR_GPU)
#include <tbb/info.h>
#endif

namespace mgis::gpu {

  // timing policy: set to true to enable warmup + timing measurement
  constexpr bool is_timed = true;

  struct thousand_sep : std::numpunct<char> {
    char do_thousands_sep() const override { return '\''; }
    std::string do_grouping() const override { return "\3"; }
  };

  template <typename T>
  std::string format_number(T value) {
    std::ostringstream oss;
    oss.imbue(std::locale(oss.getloc(), new thousand_sep));
    if constexpr (std::is_floating_point_v<T>) {
      oss << std::fixed << std::setprecision(2);
    }
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

#ifdef MGIS_GPU_HAS_CUDA_SUPPORT
  bool cuda_kernel(std::span<mgis::real>,
                   std::span<const real>,
                   const std::size_t);
#endif /* MGIS_GPU_HAS_CUDA_SUPPORT */

  using KernelType = bool (*)(std::span<mgis::real>,
                              std::span<const real>,
                              const std::size_t);

  template <bool IsTimed, bool UseGpuTiming>
  bool execute(const KernelType kernel,
               const std::size_t n,
               std::string_view program,
               std::string_view kernel_name) {
    auto F_values = std::vector<real>(9 * n, real{});
    for (std::vector<real>::size_type i = 0; i != n; ++i) {
      F_values[i] = F_values[n + i] = F_values[2 * n + i] = 1;
    }
    auto sig_values = std::vector<real>(6 * n, real{});

    if constexpr (!IsTimed) {
      return kernel(sig_values, F_values, n);
    }

    // warmup: triggers unified memory page faults + avoids lazy-instantiation
    kernel(sig_values, F_values, n);

    // timed run
    double elapsed_ms;
    bool success;

#if defined(_NVHPC_STDPAR_GPU)
    if constexpr (UseGpuTiming) {
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start);
      success = kernel(sig_values, F_values, n);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      float gpu_ms;
      cudaEventElapsedTime(&gpu_ms, start, stop);
      elapsed_ms = gpu_ms;
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
    } else
#endif
    {
      const auto start = std::chrono::steady_clock::now();
      success = kernel(sig_values, F_values, n);
      const auto end = std::chrono::steady_clock::now();
      elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    }

    std::cout << program << " " << kernel_name << " kernel for "
              << format_number(n) << " integration points: "
              << format_number(elapsed_ms) << " ms\n";
    return success;
  }  // end of execute

#ifdef MGIS_GPU_HAS_CUDA_SUPPORT
  std::span<real> allocate(const std::size_t n) {
    real *ptr;
    cudaMalloc(&ptr, n * sizeof(real));
    return std::span<real>(ptr, n);
  }

  void deallocate(std::span<real> s) {
    if (!s.empty()) {
      cudaFree(s.data());
    }
  }

  template <bool IsTimed>
  bool cuda_execute(const KernelType kernel,
                    const std::size_t n,
                    std::string_view program,
                    std::string_view kernel_name) {
    auto F_values = allocate(9 * n);
    auto sig_values = allocate(6 * n);

    // Initialize F to identity on device
    std::vector<real> F_host(9 * n, real{});
    for (std::size_t i = 0; i != n; ++i) {
      F_host[i] = F_host[n + i] = F_host[2 * n + i] = 1;
    }
    cudaMemcpy(F_values.data(), F_host.data(), 9 * n * sizeof(real), cudaMemcpyHostToDevice);

    if constexpr (!IsTimed) {
      auto success = kernel(sig_values, F_values, n);
      deallocate(F_values);
      deallocate(sig_values);
      return success;
    }

    // warmup
    kernel(sig_values, F_values, n);

    // timed run with CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    auto success = kernel(sig_values, F_values, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << program << " " << kernel_name << " kernel for "
              << format_number(n) << " integration points: "
              << format_number(static_cast<double>(gpu_ms)) << " ms\n";

    deallocate(F_values);
    deallocate(sig_values);
    return success;
  }
#endif /* MGIS_GPU_HAS_CUDA_SUPPORT */

}  // namespace mgis::gpu

int main() {
  auto success = true;
  constexpr std::size_t n = 1'000'000;
  success = mgis::gpu::execute<mgis::gpu::is_timed, false>(
                mgis::gpu::sequential_kernel, n, "signorini", "sequential") &&
            success;
#ifdef MGIS_HAS_STL_PARALLEL_ALGORITHMS
#ifdef _NVHPC_STDPAR_GPU
  constexpr auto stlpar_name = "stlpar-gpu";
  constexpr bool use_gpu_timing = true;
#else
  std::cout << "TBB threads: " << tbb::info::default_concurrency() << "\n";
  constexpr auto stlpar_name = "stlpar-cpu";
  constexpr bool use_gpu_timing = false;
#endif
  success = mgis::gpu::execute<mgis::gpu::is_timed, use_gpu_timing>(
                mgis::gpu::stlpar_kernel, n, "signorini", stlpar_name) &&
            success;
#endif /* MGIS_HAS_STL_PARALLEL_ALGORITHMS */
#ifdef MGIS_GPU_HAS_CUDA_SUPPORT
  success = mgis::gpu::cuda_execute<mgis::gpu::is_timed>(
                mgis::gpu::cuda_kernel, n, "signorini", "cuda") &&
            success;
#endif /* MGIS_GPU_HAS_CUDA_SUPPORT */
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
