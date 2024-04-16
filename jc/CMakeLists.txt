add_executable(jc gdm.cpp iomatlab.cpp jc.cpp main.cpp observer.cpp sampler.cpp)

if(MSVC)
    target_compile_definitions(jc PRIVATE "NOMINMAX" "TBB_SUPPRESS_DEPRECATED_MESSAGES")
    target_compile_options(jc PRIVATE /arch:AVX512)
else()
    target_compile_definitions(jc PRIVATE "TBB_SUPPRESS_DEPRECATED_MESSAGES")
    target_compile_options(jc PRIVATE -O3 -ffast-math -mfma -march=native -mprefer-vector-width=256)
endif()

find_package(TBB CONFIG)
if(TBB_FOUND)
    target_link_libraries(jc PUBLIC TBB::tbb)
else()
    target_link_libraries(jc PRIVATE tbb)
    target_link_options(jc PRIVATE "-L/software/software/tbb/2020.3-GCCcore-10.2.0/lib64/")
endif()

install(TARGETS jc RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)
