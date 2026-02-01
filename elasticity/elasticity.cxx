/*!
 * \file   elasticity.cxx
 * \brief
 * \author Thomas Helfer
 * \date   20/01/2026
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
  bool cuda_kernel(std::span<mgis::real>, std::span<const real>,
                   const std::size_t);
#endif /* MGIS_GPU_HAS_CUDA_SUPPORT */

  using KernelType = bool (*)(std::span<mgis::real>,
                              std::span<const real>,
                              const std::size_t);

  template <bool IsTimed, bool UseGpuTiming>
  bool execute(const KernelType kernel,
               const std::size_t n,
               std::string_view program,
               std::string_view kernel_name,
               int num_threads = 0) {
    const auto eto_values = std::vector<real>(6 * n, real{});
    auto sig_values = std::vector<real>(6 * n, real{});

    if constexpr (!IsTimed) {
      return kernel(sig_values, eto_values, n);
    }

    // warmup: triggers unified memory page faults + avoids lazy-instantiation
    kernel(sig_values, eto_values, n);

    // timed run
    double elapsed_ms;
    bool success;

#if defined(_NVHPC_STDPAR_GPU)
    if constexpr (UseGpuTiming) {
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start);
      success = kernel(sig_values, eto_values, n);
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
      success = kernel(sig_values, eto_values, n);
      const auto end = std::chrono::steady_clock::now();
      elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    }

    std::cout << program << " " << kernel_name << " kernel";
    if (num_threads > 0) {
      std::cout << " with " << num_threads << " TBB threads";
    }
    std::cout << " for "
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

  struct CudaTimer {
    cudaEvent_t start, stop;

    CudaTimer() {
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
    }

    ~CudaTimer() {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
    }

    void begin() {
      cudaEventRecord(start);
    }

    float end() {
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      float ms;
      cudaEventElapsedTime(&ms, start, stop);
      return ms;
    }
  };

  template <bool IsTimed>
  bool cuda_execute(const KernelType kernel,
                    const std::size_t n,
                    std::string_view program,
                    std::string_view kernel_name) {
    // 1. Allocate GPU buffers
    const auto eto_values = allocate(6 * n);
    auto sig_values = allocate(6 * n);

    // 2. Prepare host data
    std::vector<real> eto_host(6 * n, double(0));

    if constexpr (!IsTimed) {
      cudaMemcpy(eto_values.data(), eto_host.data(), 6 * n * sizeof(real), cudaMemcpyHostToDevice);
      auto success = kernel(sig_values, eto_values, n);
      deallocate(eto_values);
      deallocate(sig_values);
      return success;
    }

    // 3. Warmup (H2D + kernel + D2H)
    cudaMemcpy(eto_values.data(), eto_host.data(), 6 * n * sizeof(real), cudaMemcpyHostToDevice);
    kernel(sig_values, eto_values, n);
    std::vector<real> sig_warmup(6 * n);
    cudaMemcpy(sig_warmup.data(), sig_values.data(), 6 * n * sizeof(real), cudaMemcpyDeviceToHost);

    // 4. Timed section
    CudaTimer h2d_timer, kernel_timer, d2h_timer;

    // H2D
    h2d_timer.begin();
    cudaMemcpy(eto_values.data(), eto_host.data(), 6 * n * sizeof(real), cudaMemcpyHostToDevice);
    float h2d_ms = h2d_timer.end();

    // Kernel
    kernel_timer.begin();
    auto success = kernel(sig_values, eto_values, n);
    float kernel_ms = kernel_timer.end();

    // D2H
    std::vector<real> sig_host(6 * n);
    d2h_timer.begin();
    cudaMemcpy(sig_host.data(), sig_values.data(), 6 * n * sizeof(real), cudaMemcpyDeviceToHost);
    float d2h_ms = d2h_timer.end();

    // Report
    float transfer_ms = h2d_ms + d2h_ms;
    float total_ms = h2d_ms + kernel_ms + d2h_ms;

    std::cout << program << " " << kernel_name << ": "
              << format_number(static_cast<double>(kernel_ms)) << " ms kernel, "
              << format_number(static_cast<double>(transfer_ms)) << " ms transfers ("
              << format_number(static_cast<double>(h2d_ms)) << " ms H2D, "
              << format_number(static_cast<double>(d2h_ms)) << " ms D2H), "
              << format_number(static_cast<double>(total_ms)) << " ms total\n";

    deallocate(eto_values);
    deallocate(sig_values);
    return success;
  }
#endif /* MGIS_GPU_HAS_CUDA_SUPPORT */

}  // namespace mgis::gpu

int main() {
  auto success = true;
  constexpr std::size_t n = 1'000'000;
  success = mgis::gpu::execute<mgis::gpu::is_timed, false>(
                mgis::gpu::sequential_kernel, n, "elasticity", "sequential") &&
            success;
#ifdef MGIS_HAS_STL_PARALLEL_ALGORITHMS
#ifdef _NVHPC_STDPAR_GPU
  constexpr auto stlpar_name = "stlpar-gpu";
  constexpr bool use_gpu_timing = true;
  constexpr int num_threads = 0;
#else
  constexpr auto stlpar_name = "stlpar-cpu";
  constexpr bool use_gpu_timing = false;
  const int num_threads = tbb::info::default_concurrency();
#endif
  success = mgis::gpu::execute<mgis::gpu::is_timed, use_gpu_timing>(
                mgis::gpu::stlpar_kernel, n, "elasticity", stlpar_name, num_threads) &&
            success;
#endif /* MGIS_HAS_STL_PARALLEL_ALGORITHMS */
#ifdef MGIS_GPU_HAS_CUDA_SUPPORT
  success = mgis::gpu::cuda_execute<mgis::gpu::is_timed>(
                mgis::gpu::cuda_kernel, n, "elasticity", "cuda") &&
            success;
#endif /* MGIS_GPU_HAS_CUDA_SUPPORT */
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
