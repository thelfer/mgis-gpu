add_executable(DataTest DataTest.cxx)
target_link_libraries(DataTest mgis-gpu::MGISGPU)
add_test(NAME DataTest
         COMMAND DataTest ${CMAKE_CURRENT_SOURCE_DIR}/HDF5Test.h5)
