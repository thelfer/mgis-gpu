add_executable(elasticity elasticity.cxx kernels.cxx)
target_link_libraries(elasticity PRIVATE mgis::MFrontGenericInterface)

add_executable(elasticity2 elasticity2.cxx kernels2.cxx)
target_link_libraries(elasticity2 PRIVATE mgis::MFrontGenericInterface)
