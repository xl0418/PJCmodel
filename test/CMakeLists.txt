add_executable(test main.cpp)

if(MSVC)
    target_compile_definitions(test PRIVATE "NOMINMAX" "TBB_SUPPRESS_DEPRECATED_MESSAGES")
    target_compile_options(test PRIVATE /arch:AVX512)
else()
    target_compile_definitions(test PRIVATE "TBB_SUPPRESS_DEPRECATED_MESSAGES")
    target_compile_options(test PRIVATE -O3 -ffast-math -mfma -march=native -mprefer-vector-width=256)
endif()

find_package(TBB CONFIG REQUIRED)
target_link_libraries(test PRIVATE TBB::tbb)

target_include_directories(test PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../jc/")

install(TARGETS test DESTINATION ${CMAKE_SOURCE_DIR}/bin/)
