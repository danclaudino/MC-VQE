#include "PauliOperator.hpp"
#include "mc-vqe.hpp"
#include "xacc.hpp"
#include <memory>
#include <unordered_map>
#include <vector>

using namespace xacc;
using namespace xacc::quantum;

namespace xacc {
namespace algorithm {

void MC_VQE::minimizeGradients(HeterogeneousMap &parameters, std::shared_ptr<AcceleratorBuffer> buffer) {

  // initialize MC-VQE
  initialize(parameters);

  // execute
  execute(buffer);

  // optimized parameters for the current geometry
  auto x = buffer->getInformation("opt-params").as<std::vector<double>>();

  // get unrelaxed 1PDM
  auto unrelaxed1PDM = getUnrelaxed1PDM(x);

  // get unrelaxed 2PDM
  auto unrelaxed2PDM = getUnrelaxed2PDM(x);

  // get VQE Multipliers
  auto vqeMultipliers = getVQEMultipliers(x);

  // get VQE 1PDM
  auto vqe1PDM = getVQE1PDM(x, vqeMultipliers);

  // get VQE 2PDM
  auto vqe2PDM = getVQE2PDM(x, vqeMultipliers);

  // get CP-CRS multipliers
  auto crsMultipliers = getCRSMultipliers(x, vqeMultipliers);

  // get CRS 1PDM
  auto crs1PDM = getCRS1PDM(crsMultipliers); 

  // get CRS 2PDM
  auto crs2PDM = getCRS2PDM(crsMultipliers); 

  // get total 1PDM
  std::unordered_map<std::string, std::vector<Eigen::VectorXd>> total1PDM;
  for (auto term : {"X", "Z"}) {
    std::vector<Eigen::VectorXd> termTotal1PDM;
    for (int state = 0; state < nStates; state++) {
      Eigen::VectorXd total;
      total = unrelaxed1PDM[term][state] + vqe1PDM[term] + crs1PDM[term][state];
      termTotal1PDM.push_back(total);
    }
    total1PDM.emplace(term, termTotal1PDM);
  }

  // get total 2PDM
  std::unordered_map<std::string, std::vector<Eigen::MatrixXd>> total2PDM;
  for (auto term : {"XX", "XZ", "ZX", "ZZ"}) {
    std::vector<Eigen::MatrixXd> termTotal2PDM;
    for (int state = 0; state < nStates; state++) {
      Eigen::MatrixXd total;
      total = unrelaxed2PDM[term][state] + vqe2PDM[term] + crs2PDM[term][state];
      termTotal2PDM.push_back(total);
    }
    total2PDM.emplace(term, termTotal2PDM);
  }

  // get monomer gradients
  auto monomerGradients = getMonomerGradient(total1PDM);

  // get dimer interaction gradients
  auto dimerInteractionGradients = getDimerInteractionGradient(total2PDM);


  return;
}

} // namespace algorithm
} // namespace xacc
