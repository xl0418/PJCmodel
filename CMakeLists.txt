# CMakeList.txt : CMake project for jc2, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.16)

project ("jc2" CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "..." FORCE)
endif()

add_subdirectory(jc)
add_subdirectory(jcpp)
#add_subdirectory(test)
