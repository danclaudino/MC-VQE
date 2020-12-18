#include "xacc.hpp"
#include "Utils.hpp"
#include <fstream>
#include <sstream>
#include <random>

int main(int argc, char **argv) {
  xacc::Initialize(argc, argv);

  std::vector<std::string> arguments(argv + 1, argv + argc);
  int n_virt_qpus = 1, exatnLogLevel = 2, mcvqeLogLevel = 2, n_chromophores = 4,
      exatnBufferSize = 4, opt_maxiter = 1, n_states = 1, n_cycles = 1;
  std::string acc = "tnqvm", data_path;
  bool double_depth = false;
  xacc::set_verbose(true);

  for (int i = 0; i < arguments.size(); i++) {
    if (arguments[i] == "--n-chromophores") {
      n_chromophores = std::stoi(arguments[i + 1]);
    }
    if (arguments[i] == "--n-virtual-qpus") {
      n_virt_qpus = std::stoi(arguments[i + 1]);
    }
    if (arguments[i] == "--exatn-log-level") {
      exatnLogLevel = std::stoi(arguments[i + 1]);
    }
    if (arguments[i] == "--mcvqe-log-level") {
      mcvqeLogLevel = std::stoi(arguments[i + 1]);
    }
    if (arguments[i] == "--exatn-buffer-size") {
      exatnBufferSize = std::stoi(arguments[i + 1]);
    }
    if (arguments[i] == "--opt-maxiter") {
      opt_maxiter = std::stoi(arguments[i + 1]);
    }
    if (arguments[i] == "--n-states") {
      n_states = std::stoi(arguments[i + 1]);
    }
    if (arguments[i] == "--n-cycles") {
      n_cycles = std::stoi(arguments[i + 1]);
    }
    if (arguments[i] == "--accelerator") {
      acc = arguments[i + 1];
    }
    if (arguments[i] == "--data-path") {
      data_path = arguments[i + 1];
    }
    if (arguments[i] == "--double-depth") {
      if (arguments[i + 1] == "true") double_depth = true;
    }
  }
  xacc::setLoggingLevel(exatnLogLevel);
  
  if (data_path.empty()) {
    if (n_chromophores <= 18) {
      data_path = "@CMAKE_SOURCE_DIR@/examples/18_qubit_datafile.txt";
    } else {
      data_path = "@CMAKE_SOURCE_DIR@/examples/60_qubit_datafile.txt";
    }
  }
  auto optimizer = xacc::getOptimizer("nlopt", {{"nlopt-maxeval", opt_maxiter}});

  // ExaTN visitor
  std::shared_ptr<xacc::Accelerator> accelerator;
  xacc::HeterogeneousMap hetMap;
  if (acc == "tnqvm") {
    hetMap.insert("tnqvm-visitor", "exatn");
    hetMap.insert("exatn-buffer-size-gb", exatnBufferSize);
    hetMap.insert("exp-val-by-conjugate", double_depth);
    accelerator = xacc::getAccelerator(
        "tnqvm", {{"tnqvm-visitor", "exatn"},
                  {"exatn-buffer-size-gb", exatnBufferSize},
		  {"exp-val-by-conjugate", double_depth}});
  } else if (acc == "qpp") {
    accelerator = xacc::getAccelerator("qpp");
  } else if (acc == "aer") {
    accelerator = xacc::getAccelerator("aer");
  }

  // decorate accelerator
  accelerator = xacc::getAcceleratorDecorator(
      "hpc-virtualization", accelerator, {{"n-virtual-qpus", n_virt_qpus}});
  accelerator->updateConfiguration(hetMap);

  auto mc_vqe = xacc::getAlgorithm("mc-vqe");
  mc_vqe->initialize(
      {{"accelerator", accelerator},
       {"optimizer", optimizer},
       {"interference", false}, {"n-states", n_states},
       {"data-path", data_path}, {"cyclic", true},
       {"log-level", mcvqeLogLevel}, {"tnqvm-log", true},
       {"nChromophores", n_chromophores}});

for(int i = 0; i < n_cycles; i++) {
  auto q = xacc::qalloc(n_chromophores);
  xacc::ScopeTimer timer("mpi_timing", false);
  mc_vqe->execute(q);
  auto run_time = timer.getDurationMs();

  // Print the result
  if (q->hasExtraInfoKey("rank") ? ((*q)["rank"].as<int>() == 0) : true) {
    std::cout << "Energy: "
              << q->getInformation("opt-average-energy").as<double>()
              << " \n";
    std::cout << "Circuit depth: " << q->getInformation("circuit-depth").as<int>() << ".\n";
    std::cout << "Total number of gates: " << q->getInformation("n-gates").as<int>() << ".\n";
    std::cout << "Runtime: " << run_time << " ms.\n";
  }
}

  ///
  xacc::Finalize();

  return 0;
}
