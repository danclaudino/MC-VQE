/*******************************************************************************
 * Copyright (c) 2021 UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * and Eclipse Distribution License v1.0 which accompanies this
 * distribution. The Eclipse Public License is available at
 * http://www.eclipse.org/legal/epl-v10.html and the Eclipse Distribution
 *License is available at https://eclipse.org/org/documents/edl-v10.php
 *
 * Contributors:
 *   Daniel Claudino - initial API and implementation
 ******************************************************************************/
#include <gtest/gtest.h>
#include <utility>

#include "Algorithm.hpp"
#include "Optimizer.hpp"
#include "xacc.hpp"

using namespace xacc;

/*
TEST(MCVQETester, checkMCVQE) {

  std::vector<std::string> arguments(argv + 1, argv + argc);
  int exatnLogLevel = 2, mcvqeLogLevel = 1, n_chromophores = 4, n_states = 2,
      n_cycles = 1;

  // xacc::set_verbose(true);
  xacc::setLoggingLevel(exatnLogLevel);

  // path to data file
  auto data_path = "@CMAKE_SOURCE_DIR@/examples/18_qubit_datafile.txt";

  // optimizer set for a single optimization iteration
  auto optimizer = xacc::getOptimizer(
      "nlopt", {{"algorithm", "cobyla"}, {"nlopt-maxeval", 10000}});

  // ExaTN visitor
  xacc::HeterogeneousMap hetMap;
  hetMap.insert("tnqvm-visitor", "exatn");
  hetMap.insert("exatn-buffer-size-gb", 2);
  hetMap.insert("exp-val-by-conjugate", true);
  hetMap.insert("sim-type", "statevector");

  auto accelerator = xacc::getAccelerator("qpp", hetMap);

  // decorate accelerator
  accelerator = xacc::getAcceleratorDecorator("hpc-virtualization", accelerator,
                                              {{"n-virtual-qpus", 2}});
  accelerator->updateConfiguration(hetMap);

  auto mc_vqe = xacc::getAlgorithm("mc-vqe");
  mc_vqe->initialize({{"accelerator", accelerator},
                      {"optimizer", optimizer},
                      {"interference", false},
                      {"n-states", n_states},
                      {"energy-data-path", data_path},
                      {"cyclic", false},
                      {"angstrom-to-au", true},
                      {"debye-to-au", true},
                      {"log-level", mcvqeLogLevel},
                      {"tnqvm-log", true},
                      {"nChromophores", n_chromophores}});
  auto q = xacc::qalloc(n_chromophores);

  mc_vqe->execute(q);

  EXPECT_NEAR(-0.11018775, q->getInformation("opt-average-energy").as<double>(),
              1e-4);
}
*/
TEST(MCVQETester, checkSubspace) {

  std::vector<std::string> arguments(argv + 1, argv + argc);
  int exatnLogLevel = 2, mcvqeLogLevel = 2, n_chromophores = 4, n_states = 2,
      n_cycles = 1;

  // xacc::set_verbose(true);
  xacc::setLoggingLevel(exatnLogLevel);

  // path to data file
  auto data_path = "@CMAKE_SOURCE_DIR@/examples/18_qubit_datafile.txt";
  auto optimizer = xacc::getOptimizer("nlopt", {{"algorithm", "cobyla"},
                                                {"nlopt-maxeval", 10000},
                                                {"nlopt-maxeval", 1000}});

  // ExaTN visitor
  xacc::HeterogeneousMap hetMap;
  hetMap.insert("tnqvm-visitor", "exatn");
  hetMap.insert("exatn-buffer-size-gb", 2);
  hetMap.insert("exp-val-by-conjugate", true);
  hetMap.insert("sim-type", "statevector");

  auto accelerator = xacc::getAccelerator("qpp", hetMap);

  // decorate accelerator
  accelerator = xacc::getAcceleratorDecorator("hpc-virtualization", accelerator,
                                              {{"n-virtual-qpus", 2}});
  accelerator->updateConfiguration(hetMap);

  auto mc_vqe = xacc::getAlgorithm("mc-vqe");
  mc_vqe->initialize({{"accelerator", accelerator},
                      {"optimizer", optimizer},
                      {"interference", true},
                      {"n-states", n_states},
                      {"energy-data-path", data_path},
                      {"cyclic", false},
                      {"log-level", mcvqeLogLevel},
                      {"tnqvm-log", true},
                      {"angstrom-to-au", true},
                      {"debye-to-au", true},
                      {"nChromophores", n_chromophores}});
  auto q = xacc::qalloc(n_chromophores);

  std::vector<double> x = {-0.0045228, -0.0271437,   0.0269078,   -0.0318577,
                           0.0190671,  -0.000120601, -0.00932309, 0.0437225,
                           0.0138395,  0.0174245,    -0.0429059,  0.0548143,
                           -0.0347927, -0.00799183,  0.0258266,   -0.0137459};
  EXPECT_NEAR(-0.14398, mc_vqe->execute(q, x)[0], 1e-3);
}

TEST(MCVQETester, checkResponse) {

  std::vector<std::string> arguments(argv + 1, argv + argc);
  int exatnLogLevel = 2, mcvqeLogLevel = 2, n_chromophores = 2, n_states = 2,
      n_cycles = 1;

  // xacc::set_verbose(true);
  xacc::setLoggingLevel(exatnLogLevel);

  // path to data file
  auto energy_data_path = "@CMAKE_SOURCE_DIR@/examples/2_qubit_datafile.txt";
  auto response_data_path =
      "@CMAKE_SOURCE_DIR@/examples/2_qubit_datafile_response.txt";
  auto optimizer = xacc::getOptimizer(
      "nlopt", {{"algorithm", "cobyla"}, {"nlopt-maxeval", 10000}});

  // ExaTN visitor
  xacc::HeterogeneousMap hetMap;
  hetMap.insert("tnqvm-visitor", "exatn");
  hetMap.insert("exatn-buffer-size-gb", 2);
  hetMap.insert("exp-val-by-conjugate", true);

  auto accelerator = xacc::getAccelerator("qpp", hetMap);

  // decorate accelerator
  accelerator = xacc::getAcceleratorDecorator("hpc-virtualization", accelerator,
                                              {{"n-virtual-qpus", 2}});
  accelerator->updateConfiguration(hetMap);

  auto mc_vqe = xacc::getAlgorithm("mc-vqe");
  mc_vqe->initialize({{"accelerator", accelerator},
                      {"optimizer", optimizer},
                      {"interference", true},
                      {"n-states", n_states},
                      {"energy-data-path", energy_data_path},
                      {"response-data-path", response_data_path},
                      {"cyclic", false},
                      {"angstrom-to-au", false},
                      {"debye-to-au", false},
                      {"nuclear-gradient", true},
                      {"log-level", mcvqeLogLevel},
                      {"tnqvm-log", true},
                      {"entangler", "Ry"},
                      {"nChromophores", n_chromophores}});
  auto q = xacc::qalloc(n_chromophores);

  std::vector<double> x = {-0.0863268, 0.131845};
  EXPECT_NEAR(-0.0858046, mc_vqe->execute(q, x)[0], 1e-3);
}

int main(int argc, char **argv) {
  xacc::Initialize(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}