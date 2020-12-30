#include "PauliOperator.hpp"
#include "mc-vqe.hpp"
#include "xacc.hpp"
#include <memory>
#include <vector>

using namespace xacc;
using namespace xacc::quantum;

namespace xacc {
namespace algorithm {

std::vector<Eigen::VectorXd>
MC_VQE::getUnrelaxed1PDM(const std::string termStr,
                         const std::vector<double> x) {

  Eigen::MatrixXd rotatedEigenstates = CISEigenstates * subSpaceRotation;
  Eigen::MatrixXd gateAngles = statePreparationAngles(rotatedEigenstates);

  std::vector<Eigen::VectorXd> unrelaxed1PDM;
  for (int state = 0; state < nStates; state++) {

    // prepare interference state and append entangler
    auto kernel = statePreparationCircuit(gateAngles.col(state));
    kernel->addVariables(entangler->getVariables());
    for (auto &inst : entangler->getInstructions()) {
      kernel->addInstruction(inst);
    }

    Eigen::VectorXd stateDensityMatrix = Eigen::VectorXd::Zero(nChromophores);
    for (int A = 0; A < nChromophores; A++) {
      auto term = PauliOperator({{A, termStr}});
      stateDensityMatrix(A) =
          vqeWrapper(std::make_shared<PauliOperator>(term), kernel, x);
    }

    unrelaxed1PDM.push_back(stateDensityMatrix);
  }

  return unrelaxed1PDM;
}

std::vector<Eigen::MatrixXd>
MC_VQE::getUnrelaxed2PDM(const std::string termStr,
                         const std::vector<double> x) {

  Eigen::MatrixXd rotatedEigenstates = CISEigenstates * subSpaceRotation;
  Eigen::MatrixXd gateAngles = statePreparationAngles(rotatedEigenstates);

  std::vector<Eigen::MatrixXd> unrelaxed2PDM;
  for (int state = 0; state < nStates; state++) {

    // prepare interference state and append entangler
    auto kernel = statePreparationCircuit(gateAngles.col(state));
    kernel->addVariables(entangler->getVariables());
    for (auto &inst : entangler->getInstructions()) {
      kernel->addInstruction(inst);
    }

    Eigen::VectorXd stateDensityMatrix = Eigen::VectorXd::Zero(nChromophores);
    for (int A = 0; A < nChromophores; A++) {
      for (int B : pairs[A]) {
        std::string term1 = {termStr[0]}, term2 = {termStr[1]};
        auto term = PauliOperator({{A, term1}, {B, term2}});
        stateDensityMatrix(A, B) =
            vqeWrapper(std::make_shared<PauliOperator>(term), kernel, x);
      }
    }

    unrelaxed2PDM.push_back(stateDensityMatrix);
  }

  return unrelaxed2PDM;
}

void MC_VQE::getVQEMultipliers(const std::vector<double> x) {

  Eigen::MatrixXd rotatedEigenstates = CISEigenstates * subSpaceRotation;
  Eigen::MatrixXd gateAngles = statePreparationAngles(rotatedEigenstates);
  auto nParams = nChromophores * NPARAMSENTANGLER;
  std::vector<double> tmp_x;

  std::vector<Eigen::VectorXd> gradients;
  // compute gradient for each state energy w.r.t. to entangler parameters
  for (int state = 0; state < nStates; state++) {

    // prepare interference state and append entangler
    auto kernel = statePreparationCircuit(gateAngles.col(state));
    kernel->addVariables(entangler->getVariables());
    for (auto &inst : entangler->getInstructions()) {
      kernel->addInstruction(inst);
    }

    // Eq. 124
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

    // Eq. 124
    // diagonal terms
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

    // off-diagonal terms

    for (int g = 1; g < nParams; g++) {
      for (int gp = g; gp < nParams - 1; gp++) {

        tmp_x = x;
        tmp_x[g] += PI_4;
        tmp_x[gp] += PI_4;
        hessian(g, gp) += vqeWrapper(observable, kernel, tmp_x);

        tmp_x = x;
        tmp_x[g] += PI_4;
        tmp_x[gp] -= PI_4;
        hessian(g, gp) -= vqeWrapper(observable, kernel, tmp_x);

        tmp_x = x;
        tmp_x[g] -= PI_4;
        tmp_x[gp] += PI_4;
        hessian(g, gp) -= vqeWrapper(observable, kernel, tmp_x);

        tmp_x = x;
        tmp_x[g] -= PI_4;
        tmp_x[gp] -= PI_4;
        hessian(g, gp) += vqeWrapper(observable, kernel, tmp_x);

        hessian(gp, g) = hessian(g, gp);
      }
    }
  }

  hessian /= nStates;

  for (int state = 0; state < nStates; state++) {
    vqeMultipliers[state] =
        hessian.colPivHouseholderQr().solve(-gradients[state]);
  }

  return;
}

Eigen::VectorXd MC_VQE::getVQE1PDM(const std::string termStr,
                                   const std::vector<double> x) {

  // Eq. 128
  auto nParams = nChromophores * NPARAMSENTANGLER;
  std::vector<double> tmp_x;
  Eigen::VectorXd vqe1PDM = Eigen::VectorXd::Zero(nChromophores);
  for (int A = 0; A < nChromophores; A++) {

    auto term = std::make_shared<PauliOperator>(PauliOperator({{A, termStr}}));
    for (int state = 0; state < nStates; state++) {

      // prepare interference state and append entangler
      auto kernel = statePreparationCircuit(CISGateAngles.col(state));
      kernel->addVariables(entangler->getVariables());
      for (auto &inst : entangler->getInstructions()) {
        kernel->addInstruction(inst);
      }

      Eigen::VectorXd theta_g = vqeMultipliers[state];
      for (int g = 0; g < nParams; g++) {
        tmp_x = x;
        tmp_x[g] += PI_4;
        vqe1PDM(A) += theta_g(g) * vqeWrapper(term, kernel, tmp_x);

        tmp_x = x;
        tmp_x[g] -= PI_4;
        vqe1PDM(A) -= theta_g(g) * vqeWrapper(term, kernel, tmp_x);
      }
    }
  }
  vqe1PDM /= nStates;
  return vqe1PDM;
}

Eigen::MatrixXd MC_VQE::getVQE2PDM(const std::string termStr,
                                   const std::vector<double> x) {

  // Eq. 128
  auto nParams = nChromophores * NPARAMSENTANGLER;
  std::vector<double> tmp_x;
  Eigen::VectorXd vqe2PDM = Eigen::VectorXd::Zero(nChromophores);
  for (int A = 0; A < nChromophores; A++) {

    for (int B : pairs[A]) {

      std::string term1 = {termStr[0]}, term2 = {termStr[1]};
      auto term = std::make_shared<PauliOperator>(
          PauliOperator({{A, term1}, {B, term2}}));

      for (int state = 0; state < nStates; state++) {

        // prepare interference state and append entangler
        auto kernel = statePreparationCircuit(CISGateAngles.col(state));
        kernel->addVariables(entangler->getVariables());
        for (auto &inst : entangler->getInstructions()) {
          kernel->addInstruction(inst);
        }

        Eigen::VectorXd theta_g = vqeMultipliers[state];
        for (int g = 0; g < nParams; g++) {

          tmp_x = x;
          tmp_x[g] += PI_4;
          vqe2PDM(A) += theta_g(g) * vqeWrapper(term, kernel, tmp_x);

          tmp_x = x;
          tmp_x[g] -= PI_4;
          vqe2PDM(A) -= theta_g(g) * vqeWrapper(term, kernel, tmp_x);
        }
      }
    }
  }

  vqe2PDM /= nStates;
  return vqe2PDM;
}

void MC_VQE::getCRSMultipliers(const std::vector<double> x) {

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
  for (int mc = 0; mc < nStates; mc++) {

    Eigen::MatrixXd stateGrad = Eigen::MatrixXd::Zero(nStates, nStates);

    for (int crs = 0; crs < nStates; crs++) {

      for (int ci = 0; ci < nStates; ci++) {

        double grad = 0.0, first = 0.0;

        for (int M = 0; M < nStates - 1; M++) {

          if (M == 0) {

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

  // solve systems Eq. 136
  // std::vector<Eigen::MatrixXd> multipliers;
  for (int state = 0; state < nStates; state++) {
    cpCRSMultipliers.push_back(
        hessian[state].colPivHouseholderQr().solve(-gradients[state]));
  }
  return;
}

std::vector<Eigen::VectorXd> MC_VQE::getCRS1PDM(const std::string termStr) {

  // Compute D Eq. 138
  std::vector<Eigen::VectorXd> CRS1PDM;
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
            D += cpCRSMultipliers[state](I, crs) - sum * CISEigenstates(I, crs);
            D *= CISEigenstates(Ip, crs);

            //Eq. 137
            if (termStr == "Z" && Ip != 0) {
              D *= -2.0;
            }
          }
        }
      }
      stateDensityMatrix(A) = D;
    }

    CRS1PDM.push_back(stateDensityMatrix);
  }

  return CRS1PDM;
}

std::vector<Eigen::MatrixXd> MC_VQE::getCRS2PDM(const std::string termStr) {

  // Compute D Eq. 138
  std::vector<Eigen::MatrixXd> CRS2PDM;
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
            D += cpCRSMultipliers[state](I, crs) - sum * CISEigenstates(I, crs);
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

    CRS2PDM.push_back(stateDensityMatrix);
  }

  return CRS2PDM;
}


} // namespace algorithm
} // namespace xacc
