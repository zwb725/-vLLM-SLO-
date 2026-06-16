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

add_library(vllm_slo_proxy_support STATIC
    src/inflight_limiter.cpp
    src/proxy_metrics.cpp
    src/sse_parser.cpp
)

target_include_directories(vllm_slo_proxy_support
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(vllm_slo_proxy_support PUBLIC cxx_std_17)
target_link_libraries(vllm_slo_proxy_support PUBLIC vllm_slo_controller_core)

if(MSVC)
    target_compile_options(vllm_slo_proxy_support PRIVATE /W4)
else()
    target_compile_options(vllm_slo_proxy_support PRIVATE -Wall -Wextra -Wpedantic)
endif()

option(VLLM_SLO_ENABLE_PROXY "Build Boost.Asio/Beast HTTP SSE proxy when Boost is available" ON)
if(VLLM_SLO_ENABLE_PROXY)
    find_package(Boost QUIET)
    if(Boost_FOUND)
        add_library(vllm_slo_proxy_lib STATIC
            src/proxy_server.cpp
            src/proxy_session.cpp
        )
        target_compile_definitions(vllm_slo_proxy_lib PUBLIC
            VLLM_SLO_HAS_BOOST
            BOOST_ERROR_CODE_HEADER_ONLY
        )
        target_include_directories(vllm_slo_proxy_lib
            PUBLIC
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${Boost_INCLUDE_DIRS}
        )
        target_compile_features(vllm_slo_proxy_lib PUBLIC cxx_std_17)
        target_link_libraries(vllm_slo_proxy_lib PUBLIC vllm_slo_proxy_support)
        if(TARGET Boost::headers)
            target_link_libraries(vllm_slo_proxy_lib PUBLIC Boost::headers)
        endif()

        add_executable(vllm_slo_proxy src/proxy_main.cpp)
        target_link_libraries(vllm_slo_proxy PRIVATE vllm_slo_proxy_lib)

        foreach(proxy_target IN ITEMS vllm_slo_proxy_lib vllm_slo_proxy)
            if(MSVC)
                target_compile_options(${proxy_target} PRIVATE /W4)
            else()
                target_compile_options(${proxy_target} PRIVATE -Wall -Wextra -Wpedantic)
            endif()
        endforeach()
    else()
        message(WARNING "Boost was not found. Building proxy support tests only; vllm_slo_proxy is disabled.")
    endif()
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

    add_executable(test_sse_parser tests/test_sse_parser.cpp)
    target_link_libraries(test_sse_parser PRIVATE vllm_slo_proxy_support)

    add_executable(test_inflight_limiter tests/test_inflight_limiter.cpp)
    target_link_libraries(test_inflight_limiter PRIVATE vllm_slo_proxy_support)

    add_executable(test_proxy_metrics tests/test_proxy_metrics.cpp)
    target_link_libraries(test_proxy_metrics PRIVATE vllm_slo_proxy_support)

    foreach(test_target IN ITEMS
        test_sliding_window
        test_token_bucket
        test_admission_policy
        test_sse_parser
        test_inflight_limiter
        test_proxy_metrics
    )
        if(MSVC)
            target_compile_options(${test_target} PRIVATE /W4)
        else()
            target_compile_options(${test_target} PRIVATE -Wall -Wextra -Wpedantic)
        endif()
    endforeach()

    add_test(NAME sliding_window COMMAND test_sliding_window)
    add_test(NAME token_bucket COMMAND test_token_bucket)
    add_test(NAME admission_policy COMMAND test_admission_policy)
    add_test(NAME sse_parser COMMAND test_sse_parser)
    add_test(NAME inflight_limiter COMMAND test_inflight_limiter)
    add_test(NAME proxy_metrics COMMAND test_proxy_metrics)
endif()
