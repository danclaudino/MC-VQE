#include "Utils.hpp"
#include "xacc.hpp"
#include <fstream>
#include <random>
#include <sstream>

int main(int argc, char **argv) {
  xacc::Initialize(argc, argv);

  std::vector<std::string> arguments(argv + 1, argv + argc);
  int n_virt_qpus = 1, exatnLogLevel = 0, mcvqeLogLevel = 0, n_chromophores = 4,
      exatnBufferSize = 4, opt_maxiter = 1, n_states = 1, n_cycles = 1,
      max_qubit = 8;
  std::string acc = "tnqvm", energy_data_path, response_data_path;
  bool double_depth = false, print_tnqvm_log = false, debye2au = true,
       angstrom2au = false, cyclic = true, doInterference = true,
       doGradient = false;

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
    if (arguments[i] == "--print-tnqvm-log") {
      if (arguments[i + 1] == "true" || arguments[i + 1] == "1")
        print_tnqvm_log = true;
    }

    if (arguments[i] == "--cyclic") {
      if (arguments[i + 1] == "true" || arguments[i + 1] == "1") {
        cyclic = true;
      } else {
        cyclic = false;
      }
    }

    if (arguments[i] == "--angstrom-to-au") {
      if (arguments[i + 1] == "true" || arguments[i + 1] == "1") {
        angstrom2au = true;
      } else {
        angstrom2au = false;
      }
    }

    if (arguments[i] == "--debye-to-au") {
      if (arguments[i + 1] == "true" || arguments[i + 1] == "1") {
        debye2au = true;
      } else {
        debye2au = false;
      }
    }

    if (arguments[i] == "--interference") {
      if (arguments[i + 1] == "true" || arguments[i + 1] == "1") {
        doInterference = true;
      } else {
        doInterference = false;
      }
    }

    if (arguments[i] == "--nuclear-gradients") {
      if (arguments[i + 1] == "true" || arguments[i + 1] == "1") {
        doGradient = true;
      } else {
        doGradient = false;
      }
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
    if (arguments[i] == "--energy-data-path") {
      energy_data_path = arguments[i + 1];
    }
    if (arguments[i] == "--response-data-path") {
      response_data_path = arguments[i + 1];
    }
    if (arguments[i] == "--double-depth") {
      if (arguments[i + 1] == "true")
        double_depth = true;
    }
    if (arguments[i] == "--max-qubit") {
      max_qubit = std::stoi(arguments[i + 1]);
    }
  }

  if (!print_tnqvm_log)
    xacc::logToFile(true);
  xacc::setLoggingLevel(exatnLogLevel);

  if (energy_data_path.empty()) {
    if (n_chromophores == 2) {
      energy_data_path = "@CMAKE_SOURCE_DIR@/examples/2_qubit_datafile.txt";
    } else if (n_chromophores <= 18 && n_chromophores > 2) {
      energy_data_path = "@CMAKE_SOURCE_DIR@/examples/18_qubit_datafile.txt";
    } else {
      energy_data_path = "@CMAKE_SOURCE_DIR@/examples/60_qubit_datafile.txt";
    }
  }

  if (response_data_path.empty() && doGradient && n_chromophores == 2) {
    response_data_path = "@CMAKE_SOURCE_DIR@/examples/2_qubit_datafile_response.txt";
  }

  auto optimizer =
      xacc::getOptimizer("nlopt", {{"nlopt-maxeval", opt_maxiter}});

  // ExaTN visitor
  std::shared_ptr<xacc::Accelerator> accelerator;
  xacc::HeterogeneousMap hetMap;
  if (acc == "tnqvm") {
    hetMap.insert("tnqvm-visitor", "exatn");
    hetMap.insert("exatn-buffer-size-gb", exatnBufferSize);
    hetMap.insert("exp-val-by-conjugate", double_depth);
    hetMap.insert("max-qubit", max_qubit);
    accelerator = xacc::getAccelerator("tnqvm", hetMap);
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
  mc_vqe->initialize({{"accelerator", accelerator},
                      {"optimizer", optimizer},
                      {"interference", doInterference},
                      {"n-states", n_states},
                      {"energy-data-path", energy_data_path},
                      {"response-data-path", response_data_path},
                      {"cyclic", cyclic},
                      {"angstrom-to-au", angstrom2au},
                      {"debye-to-au", debye2au},
                      {"log-level", mcvqeLogLevel},
                      {"nuclear-gradient", doGradient},
                      {"tnqvm-log", print_tnqvm_log},
                      {"nChromophores", n_chromophores}});

  for (int i = 0; i < n_cycles; i++) {
    auto q = xacc::qalloc(n_chromophores);
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

  ///
  xacc::Finalize();

  return 0;
}
