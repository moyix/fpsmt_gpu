cmake_minimum_required(VERSION 3.14)
project(smt)


# Find CUDA
find_package(CUDA 11)
enable_language(CUDA)
set(ENV{CUDA_PATH} ${CUDA_TOOLKIT_ROOT_DIR})
set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
set(CMAKE_CUDA_ARCHETECTURES 35 50 72)

if (NOT DEFINED USEDLTO)
  set(USEDLTO ON)
endif()

message("")
message("USEDLTO                   ${USEDLTO}")
message("CUDA_VERSION              ${CUDA_VERSION}")
message("CUDA_PATH                 ${CUDA_TOOLKIT_ROOT_DIR}")
message("CMAKE_CUDA_COMPILER       ${CMAKE_CUDA_COMPILER}")
message("CMAKE_CUDA_ARCHETECTURES  ${CMAKE_CUDA_ARCHETECTURES}")
message("")

if (${CUDA_VERSION_MAJOR} EQUAL 11 AND ${USEDLTO})
    set(DLTO -dlto)
endif ()

set(NVCC_FLAGS -Wno-deprecated-gpu-targets --expt-relaxed-constexpr
        -DJFS_RUNTIME_FAILURE_CALLS_ABORT -dc
        -DCURAND=1 -DAES=2 -DCHAM=3)

# SMTLIB
add_library(smtlib STATIC
  SMTLIB/Core.cu
  SMTLIB/Logger.cu
  SMTLIB/NativeFloat.cu
  SMTLIB/Messages.cu
  SMTLIB/Float.cu
  SMTLIB/NativeBitVector.cu)
target_include_directories (smtlib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(smtlib PUBLIC cxx_std_11)
set_property(TARGET smtlib PROPERTY CUDA_ARCHITECTURES 35 50 72)
target_compile_options(smtlib PUBLIC ${NVCC_FLAGS})

# SMT
add_executable(smt-CURAND smt.cu theory.cu)
add_executable(smt-AES smt.cu theory.cu cuda_aes.cu)
add_executable(smt-CHAM smt.cu theory.cu cham.cu)


target_link_libraries(smt-CURAND LINK_PUBLIC smtlib)
target_link_libraries(smt-AES LINK_PUBLIC smtlib)
target_link_libraries(smt-CHAM LINK_PUBLIC smtlib)

set_property(TARGET smt-CURAND PROPERTY CUDA_ARCHITECTURES 35 50 72)
set_property(TARGET smt-AES PROPERTY CUDA_ARCHITECTURES 35 50 72)
set_property(TARGET smt-CHAM PROPERTY CUDA_ARCHITECTURES 35 50 72)

set_property(TARGET smt-CURAND PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET smt-AES PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET smt-CHAM PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(smt-CURAND PUBLIC ${NVCC_FLAGS} -DRNG=1)
target_compile_options(smt-AES PUBLIC ${NVCC_FLAGS} -DRNG=2)
target_compile_options(smt-CHAM PUBLIC ${NVCC_FLAGS} -DRNG=3)

# Move executable after build
add_custom_command(TARGET smt-CURAND
  POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../bin/)
add_custom_command(TARGET smt-AES
  POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../bin/)
add_custom_command(TARGET smt-CHAM
  POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../bin/)

add_custom_command(TARGET smt-CURAND
  POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:smt-CURAND> ../bin/smt-CURAND)
add_custom_command(TARGET smt-AES
  POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:smt-CURAND> ../bin/smt-AES)
add_custom_command(TARGET smt-CHAM
  POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:smt-CURAND> ../bin/smt-CHAM)
