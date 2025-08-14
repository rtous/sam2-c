#cmake -DCMAKE_CXX_COMPILER=clang -DBUILD_SHARED_LIBS=OFF -B build_macos -S . NO!


#cmake --debug-find -DBUILD_SHARED_LIBS=OFF -B build_macos -S .
cmake -DBUILD_SHARED_LIBS=OFF -B build_macos -S .
cmake --build build_macos
