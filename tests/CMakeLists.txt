# apps/test/CMakeLists.txt

project(test)

set(TEST_SOURCES
    ./nnTest.cpp
)

add_executable(test ${TEST_SOURCES})

if(BUILD_STATIC_LIB)
    target_link_libraries(test PRIVATE hddt_static_lib)
else()
    target_link_libraries(test PRIVATE hddt_shared_lib)
endif()

if(ROCM_FOUND)
    target_link_libraries(test PRIVATE hddt_rocm_shared)
elseif(CUDA_FOUND)
    target_link_libraries(test PRIVATE hddt_cuda_shared)
else()
    message(FATAL_ERROR "ROCm or CUDA is required but was not found.")
endif()

target_include_directories(test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# install(TARGETS test DESTINATION bin)