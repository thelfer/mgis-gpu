add_executable(plasticity plasticity.cxx kernels.cxx)
target_link_libraries(plasticity PRIVATE mgis::MFrontGenericInterface)

add_executable(plasticity2 plasticity2.cxx kernels2.cxx)
target_link_libraries(plasticity2 PRIVATE mgis::MFrontGenericInterface)
