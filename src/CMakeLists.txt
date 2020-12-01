# *******************************************************************************
# Copyright (c) 2019 UT-Battelle, LLC.
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the Eclipse Public License v1.0
# and Eclipse Distribution License v.10 which accompany this distribution.
# The Eclipse Public License is available at http://www.eclipse.org/legal/epl-v10.html
# and the Eclipse Distribution License is available at
# https://eclipse.org/org/documents/edl-v10.php
#
# Contributors:
#   Daniel Claudino - initial API and implementation
# *******************************************************************************/
set(LIBRARY_NAME mc-vqe)

file(GLOB HEADERS *.hpp)
file(GLOB SRC *.cpp)

usfunctiongetresourcesource(TARGET ${LIBRARY_NAME} OUT SRC)
usfunctiongeneratebundleinit(TARGET ${LIBRARY_NAME} OUT SRC)

add_library(${LIBRARY_NAME} SHARED ${SRC})

#target_include_directories(${LIBRARY_NAME} PUBLIC .)

#target_link_libraries(${LIBRARY_NAME} PUBLIC xacc CppMicroServices PRIVATE xacc-quantum-gate)

set(_bundle_name mc_vqe)
set_target_properties(${LIBRARY_NAME}
                      PROPERTIES COMPILE_DEFINITIONS
                                 US_BUNDLE_NAME=${_bundle_name}
                                 US_BUNDLE_NAME
                                 ${_bundle_name})

usfunctionembedresources(TARGET
                         ${LIBRARY_NAME}
                         WORKING_DIRECTORY
                         ${CMAKE_CURRENT_SOURCE_DIR}
                         FILES
                         manifest.json)

target_include_directories(${LIBRARY_NAME} PUBLIC . ${XACC_DIR}/include/eigen ${XACC_DIR})

target_link_libraries(${LIBRARY_NAME} PUBLIC xacc::xacc xacc::quantum_gate xacc::fermion xacc::pauli)

xacc_configure_plugin_rpath(mc-vqe)

install(FILES ${HEADERS} DESTINATION include)
install(TARGETS ${LIBRARY_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/plugins)