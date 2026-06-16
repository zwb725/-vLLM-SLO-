add_library(vllm_slo_controller_core STATIC
    src/admission_policy.cpp
    src/sliding_window.cpp
    src/token_bucket.cpp
)

target_include_directories(vllm_slo_controller_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(vllm_slo_controller_core PUBLIC cxx_std_17)

if(MSVC)
    target_compile_options(vllm_slo_controller_core PRIVATE /W4)
else()
    target_compile_options(vllm_slo_controller_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(controller_demo
    src/main.cpp
)

target_link_libraries(controller_demo PRIVATE vllm_slo_controller_core)

if(MSVC)
    target_compile_options(controller_demo PRIVATE /W4)
else()
    target_compile_options(controller_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(BUILD_TESTING AND VLLM_SLO_BUILD_TESTS)
    add_executable(test_sliding_window tests/test_sliding_window.cpp)
    target_link_libraries(test_sliding_window PRIVATE vllm_slo_controller_core)

    add_executable(test_token_bucket tests/test_token_bucket.cpp)
    target_link_libraries(test_token_bucket PRIVATE vllm_slo_controller_core)

    add_executable(test_admission_policy tests/test_admission_policy.cpp)
    target_link_libraries(test_admission_policy PRIVATE vllm_slo_controller_core)

    foreach(test_target IN ITEMS test_sliding_window test_token_bucket test_admission_policy)
        if(MSVC)
            target_compile_options(${test_target} PRIVATE /W4)
        else()
            target_compile_options(${test_target} PRIVATE -Wall -Wextra -Wpedantic)
        endif()
    endforeach()

    add_test(NAME sliding_window COMMAND test_sliding_window)
    add_test(NAME token_bucket COMMAND test_token_bucket)
    add_test(NAME admission_policy COMMAND test_admission_policy)
endif()
