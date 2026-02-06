/*!
 * \file   MGIS/GPU/Data.hxx
 * \brief
 * \author Thomas Helfer
 * \date   05/02/2026
 */

#ifndef LIB_MGIS_GPU_DATA_HXX
#define LIB_MGIS_GPU_DATA_HXX

#include <span>
#include <vector>
#include <optional>
#include "MGIS/Context.hxx"
#ifdef MGIS_HAVE_HDF5
#include "MGIS/Utilities/HDF5Forward.hxx"
#endif /* MGIS_HAVE_HDF5 */
#include "MGIS/GPU/Config.hxx"

namespace mgis::gpu::utilities {

  //! \brief options passed to the updateData function
  struct UpdateDataOptions {
    bool strided_output = true;
    //! \brief number of components of the data
    std::size_t number_of_components = 1;
  };

  //! \brief options passed to the generateData function
  struct GenerateDataOptions {
    bool strided_output = true;
    //! \brief number of components of the data
    std::size_t number_of_components = 1;
    /*!
     * \brief number of elements for which the data are generated.
     *
     * The size of the returned array is the number of elements times the number
     * of components
     */
    std::optional<std::size_t> number_of_elements = {};
  };

  /*!
   * \return the data generated from the inputs
   *
   * \param[in, out] ctx: execution context
   * \param[in] inputs: input data
   * \param[in] opts: options used to generate the data
   */
  MGIS_GPU_EXPORT [[nodiscard]] std::optional<std::vector<real>> generateData(
      Context&,
      std::span<const real>,
      const GenerateDataOptions& = {}) noexcept;
  /*!
   * \brief update the data from the inputs
   *
   * \param[in, out] ctx: execution context
   * \param[out] data: output
   * \param[in] inputs: input data
   * \param[in] opts: options used to update the data
   */
  MGIS_GPU_EXPORT [[nodiscard]] bool updateData(
      Context&,
      std::span<real>,
      std::span<const real>,
      const UpdateDataOptions& = {}) noexcept;

#ifdef MGIS_HAVE_HDF5

  /*!
   * \return the data generated from an HDF5 dataset
   *
   * \param[in, out] ctx: execution context
   * \param[in] g: HDF5 group
   * \param[in] n: name of the data set
   * \param[in] opts: options used to generate the data
   */
  MGIS_GPU_EXPORT [[nodiscard]] std::optional<std::vector<real>> generateData(
      Context&,
      const H5::Group&,
      const std::string&,
      const GenerateDataOptions& = {}) noexcept;
  /*!
   * \brief update the data from an HDF5 dataset
   *
   * \param[in, out] ctx: execution context
   * \param[in] g: HDF5 group
   * \param[in] n: name of the data set
   * \param[in] opts: options used to generate the data
   */
  MGIS_GPU_EXPORT [[nodiscard]] bool updateData(
      Context&,
      std::span<real>,
      const H5::Group&,
      const std::string&,
      const GenerateDataOptions& = {}) noexcept;

#endif /* MGIS_HAVE_HDF5 */

}  // end of namespace mgis::gpu::utilities

#endif /* LIB_MGIS_GPU_DATA_HXX */
