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

if (NOT DEFINED RNG)
    set(RNG CURAND)
endif()

message("")
message("RNG                       ${RNG}")
message("USEDLTO                   ${USEDLTO}")
message("CUDA_VERSION              ${CUDA_VERSION}")
message("CUDA_PATH                 ${CUDA_TOOLKIT_ROOT_DIR}")
message("CMAKE_CUDA_COMPILER       ${CMAKE_CUDA_COMPILER}")
message("CMAKE_CUDA_ARCHETECTURES  ${CMAKE_CUDA_ARCHETECTURES}")
message("")

if (${CUDA_VERSION_MAJOR} EQUAL 11 AND ${USEDLTO})
    set(DLTO -dlto)
endif ()

if (${RNG} STREQUAL CURAND)
    set(RNGN 1)
elseif(${RNG} STREQUAL AES )
    set(RNGN 2)
elseif(${RNG} STREQUAL CHAM )
    set(RNGN 3)
endif ()

set(NVCC_FLAGS -Wno-deprecated-gpu-targets --expt-relaxed-constexpr
        -DJFS_RUNTIME_FAILURE_CALLS_ABORT -dc
        -DRNG=${RNGN} -DCURAND=1 -DAES=2 -DCHAM=3)

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
if(${RNG} STREQUAL CURAND)
    add_executable(smt smt.cu theory.cu)
elseif(${RNG} STREQUAL AES)
    add_executable(smt smt.cu smt.h theory.h theory.cu cuda_aes.h cuda_aes.cu)
elseif(${RNG} STREQUAL CHAM)
    add_executable(smt smt.cu smt.h theory.h theory.cu cham.h cham.cu)
endif()
target_include_directories (smt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(smt LINK_PUBLIC smtlib)
set_property(TARGET smt PROPERTY CUDA_ARCHITECTURES 35 50 72)
set_property(TARGET smt PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(smt PUBLIC ${NVCC_FLAGS})

# Move executable after build
add_custom_command(TARGET smt
  POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../bin/)
add_custom_command(TARGET smt
  POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:smt> ../bin/smt-${RNG})
