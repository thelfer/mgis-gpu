add_executable(elasticity elasticity.cxx kernels.cxx)
target_link_libraries(elasticity PRIVATE mgis::MFrontGenericInterface)
