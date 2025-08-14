#TOP LEVEL CMAKE

#cmake_minimum_required(VERSION 3.12)
cmake_minimum_required(VERSION 3.16) #To avoid the error between RuntimeLibrary

project("back2black")

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(MSVC)
	message(STATUS "RUBEN: MSVC DETECTED")
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
	message(STATUS "RUBEN: NO MSVC DETECTED")
endif()


#from: https://trenki2.github.io/blog/2017/06/02/using-sdl2-with-cmake/
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#set (CMAKE_CXX_FLAGS "-lstdc++fs -std=c++17") 

add_subdirectory(onnxruntime) #not in third-party because PARENT_SCOPE visibility only one directory up
add_subdirectory(test_onnx)
add_subdirectory(test_sam2_aimol)
add_subdirectory(test_sam2_ryouchinsa)
add_subdirectory(test_sam2_ailia)
add_subdirectory(test_sam2_ailia_video)


