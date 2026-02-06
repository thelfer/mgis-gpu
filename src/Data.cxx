/*!
 * \file   Data.cxx
 * \brief
 * \author Thomas Helfer
 * \date   05/02/2026
 */

#include <cstdlib>
#include <cinttypes>
#include <algorithm>
#ifdef MGIS_HAVE_HDF5
#include "MGIS/Utilities/HDF5Support.hxx"
#endif /* MGIS_HAVE_HDF5 */
#include "MGIS/GPU/Data.hxx"

namespace mgis::gpu::utilities {

  std::optional<std::vector<real>> generateData(
      Context& ctx,
      std::span<const real> inputs,
      const GenerateDataOptions& opts) noexcept {
    if (opts.number_of_components == 0) {
      return ctx.registerErrorMessage("invalid number of components");
    }
    auto data = std::vector<real>{};
    if (opts.number_of_elements.has_value()) {
      if (opts.number_of_elements.value() == 0) {
        return data;
      }
      data.resize(opts.number_of_elements.value() * opts.number_of_components);
      if (!updateData(ctx, data, inputs,
                      {.strided_output = opts.strided_output,
                       .number_of_components = opts.number_of_components})) {
        data.clear();
      }
      return data;
    }
    data.resize(inputs.size());
    if (!updateData(ctx, data, inputs,
                    {.strided_output = opts.strided_output,
                     .number_of_components = opts.number_of_components})) {
      return {};
    }
    return data;
  }  // end of generateData

  bool updateData(Context& ctx,
                  std::span<real> data,
                  std::span<const real> inputs,
                  const UpdateDataOptions& opts) noexcept {
    const auto nc = opts.number_of_components;
    if (nc == 0) {
      return ctx.registerErrorMessage("invalid number of components");
    }
    const auto n = data.size() / nc;
    const auto q = data.size() % nc;
    if (q != 0) {
      return ctx.registerErrorMessage(
          "the size of the data is not a multiple of the number of "
          "components");
    }
    if (inputs.size() % opts.number_of_components != 0) {
      return ctx.registerErrorMessage(
          "the size of the inputs data is not a multiple of the number of "
          "components");
    }
    if (n == 0) {
      return true;
    }
    if (nc == 1) {
      // scalar case
      if (data.size() <= inputs.size()) {
        std::copy(inputs.begin(), inputs.begin() + data.size(), data.begin());
      } else {
        for (std::size_t i = 0; i != n; ++i) {
          const auto ioffset = i % inputs.size();
          data[i] = inputs[ioffset];
        }
      }
      return true;
    }
    // multiple components case
    if (data.size() <= inputs.size()) {
      if (opts.strided_output) {
        for (std::size_t i = 0; i != n; ++i) {
          for (std::size_t c = 0; c != nc; ++c) {
            data[n * c + i] = inputs[nc * i + c];
          }
        }
      } else {
        std::copy(inputs.begin(), inputs.begin() + data.size(), data.begin());
      }
    } else {
      if (opts.strided_output) {
        for (std::size_t i = 0; i != n; ++i) {
          const auto ioffset = nc * i % inputs.size();
          for (std::size_t c = 0; c != nc; ++c) {
            data[n * c + i] = inputs[ioffset + c];
          }
        }
      } else {
        for (std::size_t i = 0; i != n; ++i) {
          const auto ioffset = nc * i % inputs.size();
          for (std::size_t c = 0; c != nc; ++c) {
            data[nc * i + c] = inputs[ioffset + c];
          }
        }
      }
    }
    return true;
  }  // end of updateData

#ifdef MGIS_HAVE_HDF5

  std::optional<std::vector<real>> generateData(
      Context& ctx,
      const H5::Group& g,
      const std::string& n,
      const GenerateDataOptions& opts) noexcept {
    using namespace mgis::utilities::hdf5;
    auto inputs = std::vector<real>{};
    if (!read(ctx, inputs, g, n)){
      return {};
    }
    return generateData(ctx, inputs, opts);
  }  // end of generateData

  bool updateData(Context& ctx,
                  std::span<real> data,
                  const H5::Group& g,
                  const std::string& n,
                  const UpdateDataOptions& opts) noexcept {
    using namespace mgis::utilities::hdf5;
    auto inputs = std::vector<real>{};
    if (!read(ctx, inputs, g, n)){
      return {};
    }
    return updateData(ctx, data, inputs, opts);
  }  // end of updateData

#endif /* MGIS_HAVE_HDF5 */

}  // end of namespace mgis::gpu::utilities