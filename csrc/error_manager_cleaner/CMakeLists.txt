# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)
project(ttk_error_manager_cleaner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(ASCEND_HOME "$ENV{ASCEND_HOME_PATH}")
if(NOT ASCEND_HOME)
    set(ASCEND_HOME "$ENV{ASCEND_TOOLKIT_HOME}")
endif()
if(NOT ASCEND_HOME)
    message(FATAL_ERROR "ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME not set")
endif()

set(CANN_ARCH_DIR "${CMAKE_SYSTEM_PROCESSOR}-linux")
set(CANN_INCLUDE_DIR "${ASCEND_HOME}/${CANN_ARCH_DIR}/include")
set(CANN_PKG_INC_DIR "${ASCEND_HOME}/${CANN_ARCH_DIR}/pkg_inc")
set(CANN_LIB_DIR "${ASCEND_HOME}/lib64")

add_library(ttk_error_manager_cleaner SHARED error_manager_cleaner.cpp)

target_include_directories(ttk_error_manager_cleaner PRIVATE
    ${CANN_INCLUDE_DIR}
    ${CANN_PKG_INC_DIR}
)
target_link_directories(ttk_error_manager_cleaner PRIVATE ${CANN_LIB_DIR})
target_link_libraries(ttk_error_manager_cleaner PRIVATE error_manager)

target_compile_options(ttk_error_manager_cleaner PRIVATE -fvisibility=hidden -D_GLIBCXX_USE_CXX11_ABI=0)
target_link_options(ttk_error_manager_cleaner PRIVATE -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/export.map)
set_target_properties(ttk_error_manager_cleaner PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

install(TARGETS ttk_error_manager_cleaner LIBRARY DESTINATION ttk/lib)
