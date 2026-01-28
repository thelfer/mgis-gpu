add_executable(signorini signorini.cxx kernels.cxx)
target_link_libraries(signorini PRIVATE mgis::MFrontGenericInterface)

# add_executable(signorini2 signorini2.cxx kernels2.cxx)
# target_link_libraries(signorini2 PRIVATE mgis::MFrontGenericInterface)
