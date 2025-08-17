IMPORTANT: it requires C++ version 20.

OPTION 1

git clone https://github.com/xtensor-stack/xtl.git
git clone https://github.com/xtensor-stack/xtensor.git

And copy to third-party directory.

Need to add this to the main CMakeLists.txt (in this case sam2-c):

	set(CMAKE_CXX_STANDARD 20)        # required by xtensor
	set(CMAKE_CXX_STANDARD_REQUIRED ON)
	set(CMAKE_CXX_EXTENSIONS OFF)

Need to add this to the CMakeLists.txt of sam2-c/test_sam2_ailia_video:

	set(XTENSOR_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/xtensor/include")
	set(XTL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/xtl/include")

	target_include_directories(
	    ${EXAMPLE_TARGET} PUBLIC 
	    ${XTENSOR_INCLUDE_DIR}
	    ${XTL_INCLUDE_DIR}
	    ...


--------------------------

OPTION 2 (build separate projects. works but more complex)

1) Install XTL

git clone https://github.com/xtensor-stack/xtl.git

Comment the CMake version check from the beginning of CMakeLists.txt:

Just copy within third-party

cd xtl
cmake -DBUILD_SHARED_LIBS=OFF -B build_macos -S .
cmake --build build_macos


2) Install XTENSOR

git clone https://github.com/xtensor-stack/xtensor.git

Add this to the beginning of CMakeLists.txt:

	set(CMAKE_CXX_STANDARD 20)        # required by xtensor
	set(CMAKE_CXX_STANDARD_REQUIRED ON)
	set(CMAKE_CXX_EXTENSIONS OFF)

	list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../xtl/build_macos")

cd xtensor
cmake -DBUILD_SHARED_LIBS=OFF -B build_macos -S .
cmake --build build_macos

WARNING: Read build instructions in https://xtensor.readthedocs.io/en/latest/build-options.html
	- For Windows need /bigobj ...