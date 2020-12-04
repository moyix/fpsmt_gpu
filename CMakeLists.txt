cmake_minimum_required(VERSION 3.14)
project(smt LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHETECTURES 35 50 72)

add_library(smtlib STATIC
  SMTLIB/Core.cu
  SMTLIB/Logger.cu
  SMTLIB/NativeFloat.cu
  SMTLIB/Messages.cu
  SMTLIB/Float.cu
  SMTLIB/NativeBitVector.cu)
target_include_directories (smtlib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(smtlib PUBLIC cxx_std_11)

add_executable(smt smt.cu smt.h theory.h theory.cu aes.cu cuda_aes.h)
target_include_directories (smt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(smt LINK_PUBLIC smtlib)

set_property(TARGET smt PROPERTY CUDA_ARCHITECTURES 35 50 72)
set_property(TARGET smtlib PROPERTY CUDA_ARCHITECTURES 35 50 72)
set_property(TARGET smt PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(smt PUBLIC --expt-relaxed-constexpr -DJFS_RUNTIME_FAILURE_CALLS_ABORT -dc)
target_compile_options(smtlib PUBLIC --expt-relaxed-constexpr -DJFS_RUNTIME_FAILURE_CALLS_ABORT -dc)

add_custom_command(TARGET smt
  POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../bin/)
add_custom_command(TARGET smt
  POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:smt> ../bin/)
