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
//#include "xacc_service.hpp"

using namespace xacc;

TEST(MCVQETester, checkMCVQE) {

  std::vector<std::string> arguments(argv + 1, argv + argc);
  int exatnLogLevel = 2, mcvqeLogLevel = 2, n_chromophores = 4, n_states = 1,
      n_cycles = 1;

  xacc::set_verbose(true);
  xacc::setLoggingLevel(exatnLogLevel);

  // path to data file
  auto data_path = "@CMAKE_SOURCE_DIR@/examples/";

  // optimizer set for a single optimization iteration
  auto optimizer = xacc::getOptimizer("nlopt", {{"nlopt-maxeval", 1}});

  // ExaTN visitor
  xacc::HeterogeneousMap hetMap;
  hetMap.insert("tnqvm-visitor", "exatn");
  hetMap.insert("exatn-buffer-size-gb", 2);
  hetMap.insert("exp-val-by-conjugate", true);

  auto accelerator = xacc::getAccelerator("tnqvm", hetMap);

  // decorate accelerator
  accelerator = xacc::getAcceleratorDecorator("hpc-virtualization", accelerator,
                                              {{"n-virtual-qpus", 2}});
  accelerator->updateConfiguration(hetMap);

  auto mc_vqe = xacc::getAlgorithm("mc-vqe");
  mc_vqe->initialize({{"accelerator", accelerator},
                      {"optimizer", optimizer},
                      {"interference", false},
                      {"n-states", n_states},
                      {"data-path", data_path},
                      {"cyclic", true},
                      {"log-level", mcvqeLogLevel},
                      {"tnqvm-log", true},
                      {"nChromophores", n_chromophores}});

  auto q = xacc::qalloc(n_chromophores);
  for (int i = 0; i < 2; i++) {
    xacc::ScopeTimer timer("mpi_timing", false);
    mc_vqe->execute(q);
    auto run_time = timer.getDurationMs();

    // Print the result
    if (q->hasExtraInfoKey("rank") ? ((*q)["rank"].as<int>() == 0) : true) {
      std::cout << "Energy: "
                << q->getInformation("opt-average-energy").as<double>()
                << " \n";
      std::cout << "Circuit depth: "
                << q->getInformation("circuit-depth").as<int>() << ".\n";
      std::cout << "Total number of gates: "
                << q->getInformation("n-gates").as<int>() << ".\n";
      std::cout << "Runtime: " << run_time << " ms.\n";
    }
  }
   EXPECT_NEAR(-0.142685, q->getInformation("opt-average-energy").as<double>(),
   1e-4);
}

int main(int argc, char **argv) {
  xacc::Initialize(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}