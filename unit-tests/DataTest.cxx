/*!
 * \file   DataTest.cxx
 * \brief    
 * \author Thomas Helfer
 * \date   05/02/2026
 */

#include <cstdlib>
#include <iostream>
#include <numeric>
#include <algorithm>
#ifdef MGIS_HAVE_HDF5
#include "MGIS/Utilities/HDF5Support.hxx"
#endif /* MGIS_HAVE_HDF5 */
#include "MGIS/GPU/Data.hxx"

void test1(){
  auto ctx = mgis::Context{};
  auto inputs = std::vector<mgis::real>(6);
  std::iota(inputs.begin(), inputs.end(), mgis::real{});
  auto odata = mgis::gpu::utilities::generateData(ctx, inputs,
                                                  {.strided_output = true,
                                                   .number_of_components = 2,
                                                   .number_of_elements = 13});
  if (mgis::isInvalid(odata)) {
    std::cerr << ctx.getErrorMessage() << '\n';
    std::exit(EXIT_FAILURE);
  }
  for (const auto& v : *odata) {
    std::cout << " " << v;
  }
  std::cout << '\n';
}

#ifdef MGIS_HAVE_HDF5
void test2(const char* const filename) {
  auto ctx = mgis::Context{};
  auto file = H5::H5File(filename, H5F_ACC_RDONLY);
  auto root = file.openGroup("/");
  for (std::size_t idx = 0; idx != 3; ++idx) {
    auto ots = mgis::utilities::hdf5::openGroup(
        ctx, root, "TimeStep_" + std::to_string(idx));
    if (mgis::isInvalid(ots)) {
      std::cerr << ctx.getErrorMessage() << '\n';
      std::exit(EXIT_FAILURE);
    }
    auto ostrain = mgis::gpu::utilities::generateData(
        ctx, *ots, "gradients",
        {.number_of_components = 6, .number_of_elements = 2});
    if (mgis::isInvalid(ostrain)) {
      std::cerr << ctx.getErrorMessage() << '\n';
      std::exit(EXIT_FAILURE);
    }
    for (const auto& v : *ostrain) {
      std::cout << " " << v;
    }
    std::cout << '\n';
  }
}  // end of test2
#endif /* MGIS_HAVE_HDF5 */

int main(const int argc, const char* const* const argv) {
  if (argc != 2) {
    std::cerr << "invalid number of arguments\n";
    return EXIT_FAILURE;
  }
  test1();
#ifdef MGIS_HAVE_HDF5
  test2(argv[1]);
#endif /* MGIS_HAVE_HDF5 */
  return EXIT_SUCCESS;
}  // end of main
