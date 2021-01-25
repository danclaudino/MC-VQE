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

// compute the unrelaxed 1PDM
// Eq. 104/121
std::unordered_map<std::string, std::vector<Eigen::VectorXd>>
MC_VQE::getUnrelaxed1PDM(const std::vector<double> x) {

  Eigen::MatrixXd rotatedEigenstates = CISEigenstates * subSpaceRotation;
  Eigen::MatrixXd gateAngles = statePreparationAngles(rotatedEigenstates);
  std::unordered_map<std::string, std::vector<Eigen::VectorXd>> unrelaxed1PDM;

  // loop over the single Pauli term
  for (auto termStr : {"X", "Z"}) {

    // each element of term1PDM is a vector of density matrix elements
    // for each state given the termStr Pauli term
    std::vector<Eigen::VectorXd> term1PDM;
    // loop over the MC states
    for (int state = 0; state < nStates; state++) {

      // prepare interference state and append entangler
      auto kernel = statePreparationCircuit(gateAngles.col(state));
      kernel->addVariables(entangler->getVariables());
      for (auto &inst : entangler->getInstructions()) {
        kernel->addInstruction(inst);
      }

      // loop over chromophores and compute <term>
      // stateDensityMatrix stores 1PDM elements for each chromophore in state
      Eigen::VectorXd stateDensityMatrix = Eigen::VectorXd::Zero(nChromophores);
      for (int A = 0; A < nChromophores; A++) {
        auto term =
            std::make_shared<PauliOperator>(PauliOperator({{A, termStr}}));
        stateDensityMatrix(A) = vqeWrapper(term, kernel, x);
      }
      term1PDM.push_back(stateDensityMatrix);
    }
    unrelaxed1PDM.emplace(termStr, term1PDM);
  }

  return unrelaxed1PDM;
}

// compute the unrelaxed 2PDM
// Eq. 105/122
std::unordered_map<std::string, std::vector<Eigen::MatrixXd>>
MC_VQE::getUnrelaxed2PDM(const std::vector<double> x) {

  Eigen::MatrixXd rotatedEigenstates = CISEigenstates * subSpaceRotation;
  Eigen::MatrixXd gateAngles = statePreparationAngles(rotatedEigenstates);
  std::unordered_map<std::string, std::vector<Eigen::MatrixXd>> unrelaxed2PDM;

  // loop over the Pauli products
  for (auto termStr : {"XX", "XZ", "ZX", "ZZ"}) {

    // each element of term2PDM is a vector of density matrix elements
    // for each state given the termStr Pauli term
    std::vector<Eigen::MatrixXd> term2PDM;
    for (int state = 0; state < nStates; state++) {

      // prepare interference state and append entangler
      auto kernel = statePreparationCircuit(gateAngles.col(state));
      kernel->addVariables(entangler->getVariables());
      for (auto &inst : entangler->getInstructions()) {
        kernel->addInstruction(inst);
      }

      // loop over chromophores and compute <term>
      // stateDensityMatrix stores 1PDM elements for each chromophore in state
      Eigen::MatrixXd stateDensityMatrix =
          Eigen::MatrixXd::Zero(nChromophores, nChromophores);
      for (int A = 0; A < nChromophores; A++) {
        for (int B : pairs[A]) {
          std::string term1 = {termStr[0]}, term2 = {termStr[1]};
          auto term = std::make_shared<PauliOperator>(
              PauliOperator({{A, term1}, {B, term2}}));
          stateDensityMatrix(A, B) = vqeWrapper(term, kernel, x);
        }
      }
      term2PDM.push_back(stateDensityMatrix);
    }
    unrelaxed2PDM.emplace(termStr, term2PDM);
  }

  return unrelaxed2PDM;
}

// compute the VQE multipliers
std::vector<Eigen::VectorXd>
MC_VQE::getVQEMultipliers(const std::vector<double> x) {

  Eigen::MatrixXd rotatedEigenstates = CISEigenstates * subSpaceRotation;
  Eigen::MatrixXd gateAngles = statePreparationAngles(rotatedEigenstates);
  auto nParams = nChromophores * NPARAMSENTANGLER;
  std::vector<double> tmp_x;

  std::vector<Eigen::VectorXd> gradients;
  // compute gradient for each state energy w.r.t. to entangler parameters
  // Eq. 124
  for (int state = 0; state < nStates; state++) {

    // prepare interference state and append entangler
    auto kernel = statePreparationCircuit(gateAngles.col(state));
    kernel->addVariables(entangler->getVariables());
    for (auto &inst : entangler->getInstructions()) {
      kernel->addInstruction(inst);
    }

    Eigen::VectorXd stateDensityMatrix = Eigen::VectorXd::Zero(nParams);
    for (int g = 0; g < nParams; g++) {
      tmp_x = x;
      tmp_x[g] += PI_4;
      stateDensityMatrix(g) = vqeWrapper(observable, kernel, tmp_x);

      tmp_x = x;
      tmp_x[g] -= PI_4;
      stateDensityMatrix(g) -= vqeWrapper(observable, kernel, tmp_x);
    }

    gradients.push_back(stateDensityMatrix);
  }

  // Now compute Hessian
  Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(nParams, nParams);
  for (int state = 0; state < nStates; state++) {

    // prepare interference state and append entangler
    auto kernel = statePreparationCircuit(CISGateAngles.col(state));
    kernel->addVariables(entangler->getVariables());
    for (auto &inst : entangler->getInstructions()) {
      kernel->addInstruction(inst);
    }

    // diagonal Hessian elements
    for (int g = 0; g < nParams; g++) {

      // Eq. 125 term #1
      tmp_x = x;
      tmp_x[g] += 2.0 * PI_4;
      hessian(g, g) += vqeWrapper(observable, kernel, tmp_x);

      // Eq. 125 term #2
      hessian(g, g) += vqeWrapper(observable, kernel, x);

      // Eq. 125 term #3
      tmp_x = x;
      tmp_x[g] -= 2.0 * PI_4;
      hessian(g, g) -= vqeWrapper(observable, kernel, tmp_x);
    }

    // off-diagonal Hessian elements
    for (int g = 0; g < nParams; g++) {
      for (int gp = g + 1; gp < nParams; gp++) {

        // Eq. 126 term #1
        tmp_x = x;
        tmp_x[g] += PI_4;
        tmp_x[gp] += PI_4;
        hessian(g, gp) += vqeWrapper(observable, kernel, tmp_x);

        // Eq. 126 term #2
        tmp_x = x;
        tmp_x[g] += PI_4;
        tmp_x[gp] -= PI_4;
        hessian(g, gp) -= vqeWrapper(observable, kernel, tmp_x);

        // Eq. 126 term #3
        tmp_x = x;
        tmp_x[g] -= PI_4;
        tmp_x[gp] += PI_4;
        hessian(g, gp) -= vqeWrapper(observable, kernel, tmp_x);

        // Eq. 126 term #4
        tmp_x = x;
        tmp_x[g] -= PI_4;
        tmp_x[gp] -= PI_4;
        hessian(g, gp) += vqeWrapper(observable, kernel, tmp_x);

        hessian(gp, g) = hessian(g, gp);
      }
    }
  }
  hessian /= nStates;

  // Eq. 127
  std::vector<Eigen::VectorXd> vqeMultipliers;
  for (int state = 0; state < nStates; state++) {
    vqeMultipliers[state] =
        hessian.colPivHouseholderQr().solve(-gradients[state]);
  }

  return vqeMultipliers;
}

// compute CP-SA-VQE 1PDM
// this is a state-averaged density matrix
// hence all elements for a given state is the same
// and instead of a vector<Eigen::Vector>
// we have only an Eigen::Vector
std::unordered_map<std::string, Eigen::VectorXd>
MC_VQE::getVQE1PDM(const std::vector<double> x,
                   const std::vector<Eigen::VectorXd> vqeMultipliers) {

  // Eq. 128
  auto nParams = nChromophores * NPARAMSENTANGLER;
  std::vector<double> tmp_x;
  std::unordered_map<std::string, Eigen::VectorXd> vqe1PDM;
  for (auto termStr : {"X", "Z"}) {

    Eigen::VectorXd term1PDM = Eigen::VectorXd::Zero(nChromophores);
    for (int A = 0; A < nChromophores; A++) {

      auto term =
          std::make_shared<PauliOperator>(PauliOperator({{A, termStr}}));

      // sum over all states
      for (int state = 0; state < nStates; state++) {

        // prepare interference state and append entangler
        auto kernel = statePreparationCircuit(CISGateAngles.col(state));
        kernel->addVariables(entangler->getVariables());
        for (auto &inst : entangler->getInstructions()) {
          kernel->addInstruction(inst);
        }

        // sum over all VQE parameters
        for (int g = 0; g < nParams; g++) {
          tmp_x = x;
          tmp_x[g] += PI_4;
          term1PDM(A) +=
              vqeMultipliers[state](g) * vqeWrapper(term, kernel, tmp_x);

          tmp_x = x;
          tmp_x[g] -= PI_4;
          term1PDM(A) -=
              vqeMultipliers[state](g) * vqeWrapper(term, kernel, tmp_x);
        }
      }
    }
    // average over states
    term1PDM /= nStates;
    vqe1PDM.emplace(termStr, term1PDM);
  }
  return vqe1PDM;
}

std::unordered_map<std::string, Eigen::MatrixXd>
MC_VQE::getVQE2PDM(const std::vector<double> x,
                   const std::vector<Eigen::VectorXd> vqeMultipliers) {

  // Eq. 128
  auto nParams = nChromophores * NPARAMSENTANGLER;
  std::vector<double> tmp_x;
  std::unordered_map<std::string, Eigen::MatrixXd> vqe2PDM;
  for (auto termStr : {"XX", "XZ", "ZX", "ZZ"}) {

    Eigen::MatrixXd term2PDM =
        Eigen::MatrixXd::Zero(nChromophores, nChromophores);

    for (int A = 0; A < nChromophores; A++) {

      for (int B : pairs[A]) {

        std::string term1 = {termStr[0]}, term2 = {termStr[1]};
        auto term = std::make_shared<PauliOperator>(
            PauliOperator({{A, term1}, {B, term2}}));

        // sum over all states
        for (int state = 0; state < nStates; state++) {

          // prepare interference state and append entangler
          auto kernel = statePreparationCircuit(CISGateAngles.col(state));
          kernel->addVariables(entangler->getVariables());
          for (auto &inst : entangler->getInstructions()) {
            kernel->addInstruction(inst);
          }

          // sum over all VQE parameters
          for (int g = 0; g < nParams; g++) {

            tmp_x = x;
            tmp_x[g] += PI_4;
            term2PDM(A, B) +=
                vqeMultipliers[state](g) * vqeWrapper(term, kernel, tmp_x);

            tmp_x = x;
            tmp_x[g] -= PI_4;
            term2PDM(A, B) -=
                vqeMultipliers[state](g) * vqeWrapper(term, kernel, tmp_x);
          }
        }
      }
    }
    // average over states
    term2PDM /= nStates;
    vqe2PDM.emplace(termStr, term2PDM);
  }
  return vqe2PDM;
}

// compute multipliers w.r.t. contracted reference states
std::vector<Eigen::MatrixXd>
MC_VQE::getCRSMultipliers(const std::vector<double> x,
                          const std::vector<Eigen::VectorXd> vqeMultipliers) {

  // Jacobian Eq. 62
  auto jacobian = [](int M, int I, Eigen::VectorXd coefficients) {
    auto size = coefficients.size();
    double ret = 0.0;
    if (I >= M) {
      ret += coefficients(M) * coefficients(I) /
             (coefficients.segment(M + 1, size - M + 1).norm() *
              coefficients.segment(M, size - M + 1).squaredNorm());
    }

    if (I == M) {
      ret -= 1.0 / coefficients.segment(M + 1, size - M + 1).norm();
    }

    if (M == size - 1 && coefficients(size) < 0.0) {
      return -ret;
    } else {
      return ret;
    }
  };

  Eigen::MatrixXd rotatedEigenstates = CISEigenstates * subSpaceRotation;
  Eigen::MatrixXd gateAngles = statePreparationAngles(rotatedEigenstates);
  std::shared_ptr<CompositeInstruction> kernel;
  Eigen::VectorXd stateGateAngles;
  std::vector<Eigen::MatrixXd> gradients;

  // compute the terms separately then multiply
  // Eq. 131
  // loop over multicontracted/interference states
  for (int mc = 0; mc < nStates; mc++) {

    Eigen::MatrixXd stateGrad = Eigen::MatrixXd::Zero(nStates, nStates);

    // loop over CRSs states
    for (int crs = 0; crs < nStates; crs++) {

      // loop over CIS states
      for (int ci = 0; ci < nStates; ci++) {

        double grad = 0.0;

        // loop over CIS angles (M)
        for (int M = 0; M < nStates - 1; M++) {

          double first = 0.0;
          if (M == 0) {
            // Eq. 132

            stateGateAngles = gateAngles.col(mc);
            stateGateAngles(M) += PI_4;
            kernel = statePreparationCircuit(stateGateAngles);
            kernel->addVariables(entangler->getVariables());
            for (auto &inst : entangler->getInstructions()) {
              kernel->addInstruction(inst);
            }
            first += vqeWrapper(observable, kernel, x);

            stateGateAngles = gateAngles.col(mc);
            stateGateAngles(M) -= PI_4;
            kernel = statePreparationCircuit(stateGateAngles);
            kernel->addVariables(entangler->getVariables());
            for (auto &inst : entangler->getInstructions()) {
              kernel->addInstruction(inst);
            }
            first -= vqeWrapper(observable, kernel, x);

          } else {
            // Eq. 133

            stateGateAngles = gateAngles.col(mc);
            stateGateAngles(2 * M - 1) += PI_4;
            kernel = statePreparationCircuit(stateGateAngles);
            kernel->addVariables(entangler->getVariables());
            for (auto &inst : entangler->getInstructions()) {
              kernel->addInstruction(inst);
            }
            first -= vqeWrapper(observable, kernel, x);

            stateGateAngles = gateAngles.col(mc);
            stateGateAngles(2 * M - 1) -= PI_4;
            kernel = statePreparationCircuit(stateGateAngles);
            kernel->addVariables(entangler->getVariables());
            for (auto &inst : entangler->getInstructions()) {
              kernel->addInstruction(inst);
            }
            first += vqeWrapper(observable, kernel, x);

            stateGateAngles = gateAngles.col(mc);
            stateGateAngles(2 * M) += PI_4;
            kernel = statePreparationCircuit(stateGateAngles);
            kernel->addVariables(entangler->getVariables());
            for (auto &inst : entangler->getInstructions()) {
              kernel->addInstruction(inst);
            }
            first += vqeWrapper(observable, kernel, x);

            stateGateAngles = gateAngles.col(mc);
            stateGateAngles(2 * M) -= PI_4;
            kernel = statePreparationCircuit(stateGateAngles);
            kernel->addVariables(entangler->getVariables());
            for (auto &inst : entangler->getInstructions()) {
              kernel->addInstruction(inst);
            }
            first -= vqeWrapper(observable, kernel, x);

            first /= 2.0;
          }

          grad += first * jacobian(M, ci, rotatedEigenstates.col(mc));
        }

        stateGrad(ci, crs) = grad * subSpaceRotation(crs, mc);
      }
    }
    gradients.push_back(stateGrad);
  }

  // Eq. 134
  auto nParams = nChromophores * NPARAMSENTANGLER;
  for (int mc = 0; mc < nStates; mc++) {

    Eigen::MatrixXd stateGrad = Eigen::MatrixXd::Zero(nStates, nStates);

    for (int crs = 0; crs < nStates; crs++) {

      for (int ci = 0; ci < nStates; ci++) {

        double sum = 0.0;
        for (int g = 0; g < nParams; g++) {

          for (int M = 0; M < nStates - 1; M++) {

            if (M = 0) {
              stateGateAngles = CISGateAngles.col(crs);
              stateGateAngles(M) += 2.0 * PI_4;
              kernel = statePreparationCircuit(stateGateAngles);
              kernel->addVariables(entangler->getVariables());
              for (auto &inst : entangler->getInstructions()) {
                kernel->addInstruction(inst);
              }
              sum += vqeWrapper(observable, kernel, x);

              stateGateAngles = CISGateAngles.col(crs);
              stateGateAngles(M) -= 2.0 * PI_4;
              kernel = statePreparationCircuit(stateGateAngles);
              kernel->addVariables(entangler->getVariables());
              for (auto &inst : entangler->getInstructions()) {
                kernel->addInstruction(inst);
              }
              sum += vqeWrapper(observable, kernel, x);

              sum -= 2.0 * diagonal(crs);

            } else {

              stateGateAngles = gateAngles.col(crs);
              stateGateAngles(2 * M - 1) += 2.0 * PI_4;
              kernel = statePreparationCircuit(stateGateAngles);
              kernel->addVariables(entangler->getVariables());
              for (auto &inst : entangler->getInstructions()) {
                kernel->addInstruction(inst);
              }
              sum -= vqeWrapper(observable, kernel, x);

              stateGateAngles = gateAngles.col(crs);
              stateGateAngles(2 * M - 1) -= 2.0 * PI_4;
              kernel = statePreparationCircuit(stateGateAngles);
              kernel->addVariables(entangler->getVariables());
              for (auto &inst : entangler->getInstructions()) {
                kernel->addInstruction(inst);
              }
              sum -= vqeWrapper(observable, kernel, x);

              stateGateAngles = gateAngles.col(crs);
              stateGateAngles(2 * M) += 2.0 * PI_4;
              kernel = statePreparationCircuit(stateGateAngles);
              kernel->addVariables(entangler->getVariables());
              for (auto &inst : entangler->getInstructions()) {
                kernel->addInstruction(inst);
              }
              sum += vqeWrapper(observable, kernel, x);

              stateGateAngles = gateAngles.col(crs);
              stateGateAngles(2 * M) -= 2.0 * PI_4;
              kernel = statePreparationCircuit(stateGateAngles);
              kernel->addVariables(entangler->getVariables());
              for (auto &inst : entangler->getInstructions()) {
                kernel->addInstruction(inst);
              }
              sum += vqeWrapper(observable, kernel, x);

              sum /= 2.0;
            }

            sum *= jacobian(M, ci, gateAngles.col(crs)) * vqeMultipliers[mc](g);

          } // M loop
        }   // g loop

        gradients[mc](ci, crs) += sum / nStates;

      } // ci loop
    }   // crs loop
  }     // mc loop

  // compute Hessian Eq. 135
  std::vector<Eigen::MatrixXd> hessian;
  for (int state = 0; state < nStates; state++) {

    Eigen::MatrixXd stateHessian = Eigen::MatrixXd::Zero(nStates, nStates);
    for (int I = 0; I < nStates; I++) {
      for (int Ip = I; Ip < nStates; Ip++) {

        stateHessian(I, Ip) =
            CISHamiltonian(I, Ip) - 2.0 * CISEnergies(state) *
                                        CISEigenstates(I, state) *
                                        CISEigenstates(Ip, state);

        if (I == Ip) {
          stateHessian(I, Ip) -= CISEnergies(state);
        }
      }
    }
  }

  // Eq. 136
  std::vector<Eigen::MatrixXd> cpCRSMultipliers;
  for (int state = 0; state < nStates; state++) {
    cpCRSMultipliers.push_back(
        hessian[state].colPivHouseholderQr().solve(-gradients[state]));
  }
  return cpCRSMultipliers;
}

// compute CRS contribution to the relaxed 1PDM
std::unordered_map<std::string, std::vector<Eigen::VectorXd>>
MC_VQE::getCRS1PDM(const std::vector<Eigen::MatrixXd> cpCRSMultipliers) {

  // Compute D Eq. 138
  std::unordered_map<std::string, std::vector<Eigen::VectorXd>> CRS1PDM;

  for (auto termStr : {"X", "Z"}) {

    std::vector<Eigen::VectorXd> term1PDM;
    for (int state = 0; state < nStates; state++) {

      Eigen::VectorXd stateDensityMatrix = Eigen::VectorXd::Zero(nChromophores);
      for (int A = 0; A < nChromophores; A++) {

        double D = 0.0;
        for (int I = 0; I < nStates; I++) {

          for (int Ip = 0; Ip < nStates; Ip++) {

            if (termStr == "Z" && I != Ip)
              continue;
            if (termStr == "X" && (I != 0 && Ip != 0))
              continue;

            for (int crs = 0; crs < nStates; crs++) {
              double sum = 0.0;
              for (int i = 0; i < nStates; i++) {
                sum += cpCRSMultipliers[state](i, crs) * CISEigenstates(i, crs);
              }
              D += cpCRSMultipliers[state](I, crs) -
                   sum * CISEigenstates(I, crs);
              D *= CISEigenstates(Ip, crs);

              // Eq. 137
              if (termStr == "Z" && Ip != 0) {
                D *= -2.0;
              }
            }
          }
        }
        stateDensityMatrix(A) = D;
      }
      term1PDM.push_back(stateDensityMatrix);
    }
    CRS1PDM.emplace(termStr, term1PDM);
  }

  return CRS1PDM;
}

// compute CRS contribution to the relaxed 2PDM
std::unordered_map<std::string, std::vector<Eigen::MatrixXd>>
MC_VQE::getCRS2PDM(const std::vector<Eigen::MatrixXd> cpCRSMultipliers) {

  // Compute D Eq. 138
  std::unordered_map<std::string, std::vector<Eigen::MatrixXd>> CRS2PDM;

  for (auto termStr : {"XX", "XZ", "ZX", "ZZ"}) {

    std::vector<Eigen::MatrixXd> term2PDM;
    for (int state = 0; state < nStates; state++) {

      Eigen::MatrixXd stateDensityMatrix =
          Eigen::MatrixXd::Zero(nChromophores, nChromophores);
      for (int A = 0; A < nChromophores; A++) {

        double D = 0.0;
        for (int I = 0; I < nStates; I++) {

          for (int Ip = 0; Ip < nStates; Ip++) {

            if (termStr == "ZZ" && I != Ip)
              continue;
            if (termStr == "XX" && I == Ip)
              continue;
            if ((termStr == "ZX" || termStr == "XZ") && (I != 0 && Ip != 0))
              continue;

            for (int crs = 0; crs < nStates; crs++) {
              double sum = 0.0;
              for (int i = 0; i < nStates; i++) {
                sum += cpCRSMultipliers[state](i, crs) * CISEigenstates(i, crs);
              }
              D += cpCRSMultipliers[state](I, crs) -
                   sum * CISEigenstates(I, crs);
              D *= CISEigenstates(Ip, crs);

              // Eq. 137
              if (termStr == "ZZ" && Ip != 0) {
                D *= -2.0;
              }
              if (termStr == "ZX" || termStr == "XZ") {
                D *= 0.5;
              }
            }
          }
        }
        for (auto B : pairs[A]) {
          stateDensityMatrix(A, B) = D;
        }
      }
      term2PDM.push_back(stateDensityMatrix);
    }
    CRS2PDM.emplace(termStr, term2PDM);
  }

  return CRS2PDM;
}

std::unordered_map<std::string, std::vector<Eigen::VectorXd>>
MC_VQE::getMonomerGradient(
    std::unordered_map<std::string, std::vector<Eigen::VectorXd>> _1PDM) {

  std::unordered_map<std::string, std::vector<Eigen::VectorXd>>
      monomerGradients;

  // Eq. 143

  for (auto termStr : {"H", "P"}) {

    std::vector<Eigen::VectorXd> termGrad;

    for (int state = 0; state < nStates; state++) {

      Eigen::VectorXd grad = Eigen::VectorXd::Constant(nChromophores, 0.5);

      if (termStr == "H") {
        grad += 0.5 * _1PDM["Z"][state];
      }

      if (termStr == "P") {
        grad -= 0.5 * _1PDM["Z"][state];
      }

      termGrad.push_back(grad);
    }

    monomerGradients.emplace(termStr, termGrad);
  }

  return monomerGradients;
}

std::unordered_map<std::string, std::vector<Eigen::MatrixXd>>
MC_VQE::getDimerInteractionGradient(
    std::unordered_map<std::string, std::vector<Eigen::MatrixXd>> _2PDM) {

  // Eq. 147
  auto dipolePartial = [&](const Eigen::Vector3d mu, const Eigen::Vector3d r) {
    auto d = r.norm();
    return (mu / std::pow(d, 3.0) - 3.0 * mu.dot(r) * r / std::pow(d, 5.0));
  };

  // Eq. 148
  auto distancePartial = [&](const Eigen::Vector3d muA,
                             const Eigen::Vector3d muB,
                             const Eigen::Vector3d r) {
    auto d = r.norm();
    Eigen::Vector3d ret = Eigen::Vector3d::Zero();
    ret -= 3.0 * muA.dot(muB) / std::pow(d, 5) * r;
    ret += 15.0 * muA.dot(r) * muB.dot(r) / std::pow(d, 7) * r;
    ret -= 3.0 * muB.dot(r) / std::pow(d, 5) * muA;
    ret -= 3.0 * muA.dot(r) / std::pow(d, 5) * muB;

    return ret;
  };

  std::unordered_map<std::string, std::vector<Eigen::MatrixXd>>
      dimerInteractionGradients;

  for (auto termStr : {"H", "P", "T", "R"}) {

    std::vector<Eigen::MatrixXd> termEta;
    for (int state = 0; state < nStates; state++) {

      Eigen::MatrixXd eta = Eigen::MatrixXd::Zero(nChromophores, 3);
      for (int A = 0; A < nChromophores; A++) {

        for (auto B : pairs[A]) {

          auto r = centerOfMass.row(A) - centerOfMass.row(B);

          // Eq. 145
          if (termStr == "H") {
            eta.row(A) += dipolePartial(groundStateDipoles.row(B), r) / 4.0 *
                          _2PDM["ZZ"][state](A, B);
            eta.row(A) -= dipolePartial(excitedStateDipoles.row(B), r) / 4.0 *
                          _2PDM["ZZ"][state](A, B);
            eta.row(A) += dipolePartial(transitionDipoles.row(B), r) / 2.0 *
                          _2PDM["ZX"][state](A, B);
            eta.row(B) += dipolePartial(groundStateDipoles.row(A), r) / 4.0 *
                          _2PDM["ZZ"][state](A, B);
            eta.row(B) -= dipolePartial(excitedStateDipoles.row(A), r) / 4.0 *
                          _2PDM["ZZ"][state](A, B);
            eta.row(B) += dipolePartial(transitionDipoles.row(A), r) / 2.0 *
                          _2PDM["XZ"][state](A, B);
          }

          // Eq. 145
          if (termStr == "P") {
            eta.row(A) -= dipolePartial(groundStateDipoles.row(B), r) / 4.0 *
                          _2PDM["ZZ"][state](A, B);
            eta.row(A) += dipolePartial(excitedStateDipoles.row(B), r) / 4.0 *
                          _2PDM["ZZ"][state](A, B);
            eta.row(A) -= dipolePartial(transitionDipoles.row(B), r) / 2.0 *
                          _2PDM["ZX"][state](A, B);
            eta.row(B) -= dipolePartial(groundStateDipoles.row(A), r) / 4.0 *
                          _2PDM["ZZ"][state](A, B);
            eta.row(B) += dipolePartial(excitedStateDipoles.row(A), r) / 4.0 *
                          _2PDM["ZZ"][state](A, B);
            eta.row(B) -= dipolePartial(transitionDipoles.row(A), r) / 2.0 *
                          _2PDM["XZ"][state](A, B);
          }

          // Eq. 145
          if (termStr == "T") {
            eta.row(A) += dipolePartial(transitionDipoles.row(B), r) *
                          _2PDM["XX"][state](A, B);
            eta.row(A) += dipolePartial(groundStateDipoles.row(B), r) / 2.0 *
                          _2PDM["XZ"][state](A, B);
            eta.row(A) -= dipolePartial(excitedStateDipoles.row(B), r) / 2.0 *
                          _2PDM["XZ"][state](A, B);
            eta.row(B) += dipolePartial(transitionDipoles.row(A), r) *
                          _2PDM["XX"][state](A, B);
            eta.row(B) += dipolePartial(groundStateDipoles.row(A), r) / 2.0 *
                          _2PDM["ZX"][state](A, B);
            eta.row(B) -= dipolePartial(excitedStateDipoles.row(A), r) / 2.0 *
                          _2PDM["ZX"][state](A, B);
          }

          // Eq. 146
          if (termStr == "R") {
            eta.row(A) -= distancePartial(groundStateDipoles.row(A),
                                          groundStateDipoles.row(B), r) /
                          4.0 * _2PDM["ZZ"][state](A, B);
            eta.row(B) += distancePartial(groundStateDipoles.row(B),
                                          groundStateDipoles.row(A), r) /
                          4.0 * _2PDM["ZZ"][state](A, B);

            // HP
            eta.row(A) += distancePartial(groundStateDipoles.row(A),
                                          excitedStateDipoles.row(B), r) /
                          4.0 * _2PDM["ZZ"][state](A, B);
            eta.row(B) -= distancePartial(groundStateDipoles.row(B),
                                          excitedStateDipoles.row(A), r) /
                          4.0 * _2PDM["ZZ"][state](A, B);

            // HT
            eta.row(A) -= distancePartial(groundStateDipoles.row(A),
                                          transitionDipoles.row(B), r) /
                          2.0 * _2PDM["ZX"][state](A, B);
            eta.row(B) += distancePartial(groundStateDipoles.row(B),
                                          transitionDipoles.row(A), r) /
                          2.0 * _2PDM["ZX"][state](A, B);

            // PP
            eta.row(A) -= distancePartial(excitedStateDipoles.row(A),
                                          excitedStateDipoles.row(B), r) /
                          4.0 * _2PDM["ZZ"][state](A, B);
            eta.row(B) += distancePartial(excitedStateDipoles.row(B),
                                          excitedStateDipoles.row(A), r) /
                          4.0 * _2PDM["ZZ"][state](A, B);

            // PH
            eta.row(A) += distancePartial(excitedStateDipoles.row(A),
                                          groundStateDipoles.row(B), r) /
                          4.0 * _2PDM["ZZ"][state](A, B);
            eta.row(B) -= distancePartial(excitedStateDipoles.row(B),
                                          groundStateDipoles.row(A), r) /
                          4.0 * _2PDM["ZZ"][state](A, B);

            // PT
            eta.row(A) += distancePartial(excitedStateDipoles.row(A),
                                          transitionDipoles.row(B), r) /
                          2.0 * _2PDM["ZX"][state](A, B);
            eta.row(B) -= distancePartial(excitedStateDipoles.row(B),
                                          transitionDipoles.row(A), r) /
                          2.0 * _2PDM["ZX"][state](A, B);

            // TT
            eta.row(A) -= distancePartial(transitionDipoles.row(A),
                                          transitionDipoles.row(B), r) *
                          _2PDM["XX"][state](A, B);
            eta.row(B) += distancePartial(transitionDipoles.row(B),
                                          transitionDipoles.row(A), r) *
                          _2PDM["XX"][state](A, B);

            // TH
            eta.row(A) -= distancePartial(transitionDipoles.row(A),
                                          groundStateDipoles.row(B), r) /
                          2.0 * _2PDM["XZ"][state](A, B);
            eta.row(B) += distancePartial(transitionDipoles.row(B),
                                          groundStateDipoles.row(A), r) /
                          2.0 * _2PDM["XZ"][state](A, B);

            // TP
            eta.row(A) += distancePartial(transitionDipoles.row(A),
                                          excitedStateDipoles.row(B), r) /
                          2.0 * _2PDM["XZ"][state](A, B);
            eta.row(B) -= distancePartial(transitionDipoles.row(B),
                                          excitedStateDipoles.row(A), r) /
                          2.0 * _2PDM["XZ"][state](A, B);
          }
        }
      }
      termEta.push_back(eta);
    }
    dimerInteractionGradients.emplace(termStr, termEta);
  }
  return dimerInteractionGradients;
}

} // namespace algorithm
} // namespace xacc
