add_executable(jcpp jcpp.cpp ../jc/gdm.cpp ../jc/iomatlab.cpp)

if(MSVC)
    target_compile_definitions(jcpp PRIVATE "NOMINMAX" "TBB_SUPPRESS_DEPRECATED_MESSAGES")
    target_compile_options(jcpp PRIVATE /arch:AVX512)
else()
    target_compile_definitions(jcpp PRIVATE "TBB_SUPPRESS_DEPRECATED_MESSAGES")
    target_compile_options(jcpp PRIVATE -O3 -ffast-math -mfma -march=native -mprefer-vector-width=256)
endif()

find_package(TBB CONFIG)
if(TBB_FOUND)
    target_link_libraries(jcpp PUBLIC TBB::tbb)
else()
    target_link_libraries(jcpp PRIVATE tbb)
    target_link_options(jcpp PRIVATE "-L/software/software/tbb/2020.3-GCCcore-10.2.0/lib64/")
endif()

target_include_directories(jcpp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../jc/")

install(TARGETS jcpp DESTINATION ${CMAKE_SOURCE_DIR}/bin/)
