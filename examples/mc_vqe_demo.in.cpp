#include "Utils.hpp"
#include "xacc.hpp"
#include <fstream>
#include <random>
#include <sstream>

int main(int argc, char **argv) {
  xacc::Initialize(argc, argv);

  int exatnLogLevel = 2, mcvqeLogLevel = 2, n_states = 5,
      n_cycles = 1;
  xacc::setLoggingLevel(exatnLogLevel);

  // path to data file
  auto data_path = "@CMAKE_SOURCE_DIR@/examples/";
  auto optimizer = xacc::getOptimizer("nlopt", {{"algorithm", "cobyla"}, {"nlopt-maxeval", 10000}, {"nlopt-maxeval", 1000}});

  auto accelerator = xacc::getAccelerator("qpp");

  // get reference to Accelerator and Optimizer
  // define data_path
  auto n_chromophores = 4;
  auto mc_vqe = xacc::getAlgorithm("mc-vqe");
  mc_vqe->initialize({{"accelerator", accelerator},
                      {"optimizer", optimizer},
                      {"interference", true},
                      {"n-states", n_states},
                      {"data-path", data_path},
                      {"cyclic", true},
                      {"log-level", mcvqeLogLevel},
                      {"tnqvm-log", true},
                      {"angstrom-to-au", true},
                      {"debye-to-au", true},
                      {"nChromophores", n_chromophores}});

// allocate buffer and execute
auto buffer = xacc::qalloc(n_chromophores);
mc_vqe->execute(buffer);

  return 0;
}
