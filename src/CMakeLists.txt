set(MGISGPU_SOURCES)
list(APPEND MGISGPU_SOURCES Data.cxx)

add_library(MGISGPU SHARED ${MGISGPU_SOURCES})
add_library(mgis-gpu::MGISGPU ALIAS MGISGPU)
target_include_directories(MGISGPU
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_link_libraries(MGISGPU
        PUBLIC mgis::MFrontGenericInterface)
install(TARGETS MGISGPU EXPORT MGISGPU
  DESTINATION lib)
install(EXPORT MGISGPU DESTINATION "share/mgis-gpu/cmake"
        EXPORT_LINK_INTERFACE_LIBRARIES
        NAMESPACE mgis-gpu:: FILE MGISGPUTargets.cmake)
