cmake_minimum_required(VERSION 3.21)

project(vllm_slo_controller
    VERSION 0.1.0
    DESCRIPTION "Phase 0 C++17 core for a vLLM SLO admission controller"
    LANGUAGES CXX
)

option(VLLM_SLO_BUILD_TESTS "Build CTest based unit tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
if(BUILD_TESTING AND VLLM_SLO_BUILD_TESTS)
    enable_testing()
endif()

add_subdirectory(controller)
