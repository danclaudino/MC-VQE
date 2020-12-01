#include "mc-vqe.hpp"
#include "PauliOperator.hpp"
#include <memory>
#include <vector>
#include "xacc.hpp"

using namespace xacc;
using namespace xacc::quantum;

namespace xacc {
namespace algorithm {

std::vector<Eigen::VectorXd>
MC_VQE::getUnrelaxed1PDM(const std::string pauliTerm,
                         const std::vector<double> x) {

  Eigen::MatrixXd rotatedEigenstates = CISEigenstates * subspaceRotation;
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
      auto term = PauliOperator({{A, pauliTerm}});
      stateDensityMatrix(A) =
          vqeWrapper(std::make_shared<PauliOperator>(term), kernel, x);
    }

    unrelaxed1PDM.push_back(stateDensityMatrix);
  }

  return unrelaxed1PDM;
}

std::vector<Eigen::MatrixXd>
MC_VQE::getUnrelaxed2PDM(const std::string pauliTerm,
                         const std::vector<double> x) {

  Eigen::MatrixXd rotatedEigenstates = CISEigenstates * subspaceRotation;
  Eigen::MatrixXd gateAngles = statePreparationAngles(rotatedEigenstates);

  // stores the indices of the valid chromophore pairs
  std::vector<std::vector<int>> pairs(nChromophores);
  for (int A = 0; A < nChromophores; A++) {
    if (A == 0 && isCyclic) {
      pairs[A] = {A + 1, nChromophores - 1};
    } else if (A == nChromophores - 1) {
      pairs[A] = {A - 1, 0};
    } else {
      pairs[A] = {A - 1, A + 1};
    }
  }

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
        std::string term1 = {pauliTerm[0]}, term2 = {pauliTerm[1]};
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

  Eigen::MatrixXd rotatedEigenstates = CISEigenstates * subspaceRotation;
  Eigen::MatrixXd gateAngles = statePreparationAngles(rotatedEigenstates);
  int nParams = x.size();
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
      tmp_x[g] += xacc::constants::pi / 4.0;
      stateDensityMatrix(g) = vqeWrapper(observable, kernel, tmp_x);

      tmp_x = x;
      tmp_x[g] -= xacc::constants::pi / 4.0;
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
      tmp_x[g] += xacc::constants::pi / 2.0;
      hessian(g, g) += vqeWrapper(observable, kernel, tmp_x);

      // Eq. 125 term #2
      hessian(g, g) += vqeWrapper(observable, kernel, x);

      // Eq. 125 term #3
      tmp_x = x;
      tmp_x[g] -= xacc::constants::pi / 2.0;
      hessian(g, g) -= vqeWrapper(observable, kernel, tmp_x);
    }

    // off-diagonal terms

    for (int g = 1; g < nParams; g++) {
      for (int gp = g; gp < nParams - 1; gp++) {

        tmp_x = x;
        tmp_x[g] += xacc::constants::pi / 4.0;
        tmp_x[gp] += xacc::constants::pi / 4.0;
        hessian(g, gp) += vqeWrapper(observable, kernel, tmp_x);

        tmp_x = x;
        tmp_x[g] += xacc::constants::pi / 4.0;
        tmp_x[gp] -= xacc::constants::pi / 4.0;
        hessian(g, gp) -= vqeWrapper(observable, kernel, tmp_x);

        tmp_x = x;
        tmp_x[g] -= xacc::constants::pi / 4.0;
        tmp_x[gp] += xacc::constants::pi / 4.0;
        hessian(g, gp) -= vqeWrapper(observable, kernel, tmp_x);

        tmp_x = x;
        tmp_x[g] -= xacc::constants::pi / 4.0;
        tmp_x[gp] -= xacc::constants::pi / 4.0;
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

Eigen::VectorXd MC_VQE::getVQE1PDM(const std::string pauliTerm,
                                   const std::vector<double> x) {

  int nParams = x.size();
  std::vector<double> tmp_x;

  // Eq. 128
  Eigen::VectorXd vqe1PDM = Eigen::VectorXd::Zero(nChromophores);
  for (int A = 0; A < nChromophores; A++) {

    auto term = PauliOperator({{A, pauliTerm}});

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
        tmp_x[g] += xacc::constants::pi / 4.0;
        vqe1PDM(A) +=
            theta_g(g) *
            vqeWrapper(std::make_shared<PauliOperator>(term), kernel, tmp_x);

        tmp_x = x;
        tmp_x[g] -= xacc::constants::pi / 4.0;
        vqe1PDM(A) -=
            theta_g(g) *
            vqeWrapper(std::make_shared<PauliOperator>(term), kernel, tmp_x);
      }
    }
  }

  vqe1PDM /= nStates;

  return vqe1PDM;
}

Eigen::MatrixXd MC_VQE::getVQE2PDM(const std::string pauliTerm,
                                   const std::vector<double> x) {

  // stores the indices of the valid chromophore pairs
  std::vector<std::vector<int>> pairs(nChromophores);
  for (int A = 0; A < nChromophores; A++) {
    if (A == 0 && isCyclic) {
      pairs[A] = {A + 1, nChromophores - 1};
    } else if (A == nChromophores - 1) {
      pairs[A] = {A - 1, 0};
    } else {
      pairs[A] = {A - 1, A + 1};
    }
  }

  int nParams = x.size();
  std::vector<double> tmp_x;
  // Eq. 128
  Eigen::VectorXd vqe2PDM = Eigen::VectorXd::Zero(nChromophores);
  for (int A = 0; A < nChromophores; A++) {

    for (int B : pairs[A]) {

      std::string term1 = {pauliTerm[0]}, term2 = {pauliTerm[1]};
      auto term = PauliOperator({{A, term1}, {B, term2}});

      for (int state = 0; state < nStates; state++) {

        // prepare interference state and append entangler
        auto kernel = statePreparationCircuit(CISGateAngles.col(state));
        kernel->addVariables(entangler->getVariables());
        for (auto &inst : entangler->getInstructions()) {
          kernel->addInstruction(inst);
        }

        Eigen::VectorXd theta_g = vqeMultipliers[state];
        // hessian.colPivHouseholderQr().solve(-gradients[state]);

        for (int g = 0; g < nParams; g++) {

          tmp_x = x;
          tmp_x[g] += xacc::constants::pi / 4.0;
          vqe2PDM(A) +=
              theta_g(g) *
              vqeWrapper(std::make_shared<PauliOperator>(term), kernel, tmp_x);

          tmp_x = x;
          tmp_x[g] -= xacc::constants::pi / 4.0;
          vqe2PDM(A) -=
              theta_g(g) *
              vqeWrapper(std::make_shared<PauliOperator>(term), kernel, tmp_x);
        }
      }
    }
  }

  vqe2PDM /= nStates;

  return vqe2PDM;
}

/*
std::vector<Eigen::VectorXd> MC_VQE::getCRS1PDM(const std::string pauliTerm,
                                                const std::vector<double> x) {

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

  Eigen::MatrixXd rotatedEigenstates = CISEigenstates * subspaceRotation;
  Eigen::MatrixXd gateAngles = statePreparationAngles(rotatedEigenstates);
  std::shared_ptr<CompositeInstruction> kernel;

  std::vector<Eigen::MatrixXd> gradients;

  // compute the terms separately then multiply
  for (int mc = 0; mc < nStates; mc++) {

    for (int crs = 0; crs < nStates; crs++) {

      for (int ci = 0; ci < nStates; ci++) {

      }

    }

  }

  // Eq. 131
  for (int state = 0; state < nStates; state++) {

    double grad = 0.0, sum = 0.0;
    Eigen::VectorXd stateGrad = Eigen::VectorXd::Zero(nChromophores);
    for (int M = 0; M < nChromophores; M++) {

      double first = 0.0;
      Eigen::VectorXd stateGateAngles;

      // Eq. 132
      if (M == 0) {

        stateGateAngles = gateAngles.col(state);
        stateGateAngles(M) += xacc::constants::pi / 4.0;
        kernel = statePreparationCircuit(stateGateAngles);
        kernel->addVariables(entangler->getVariables());
        for (auto &inst : entangler->getInstructions()) {
          kernel->addInstruction(inst);
        }
        first += vqeWrapper(observable, kernel, x);

        stateGateAngles = gateAngles.col(state);
        stateGateAngles(M) -= xacc::constants::pi / 4.0;
        kernel = statePreparationCircuit(stateGateAngles);
        kernel->addVariables(entangler->getVariables());
        for (auto &inst : entangler->getInstructions()) {
          kernel->addInstruction(inst);
        }
        first -= vqeWrapper(observable, kernel, x);

      } else {
        // Eq. 133

        stateGateAngles = gateAngles.col(state);
        stateGateAngles(2 * M - 1) += xacc::constants::pi / 4.0;
        kernel = statePreparationCircuit(stateGateAngles);
        kernel->addVariables(entangler->getVariables());
        for (auto &inst : entangler->getInstructions()) {
          kernel->addInstruction(inst);
        }
        first -= vqeWrapper(observable, kernel, x);

        stateGateAngles = gateAngles.col(state);
        stateGateAngles(2 * M - 1) -= xacc::constants::pi / 4.0;
        kernel = statePreparationCircuit(stateGateAngles);
        kernel->addVariables(entangler->getVariables());
        for (auto &inst : entangler->getInstructions()) {
          kernel->addInstruction(inst);
        }
        first += vqeWrapper(observable, kernel, x);

        stateGateAngles = gateAngles.col(state);
        stateGateAngles(2 * M) += xacc::constants::pi / 4.0;
        kernel = statePreparationCircuit(stateGateAngles);
        kernel->addVariables(entangler->getVariables());
        for (auto &inst : entangler->getInstructions()) {
          kernel->addInstruction(inst);
        }
        first += vqeWrapper(observable, kernel, x);

        stateGateAngles = gateAngles.col(state);
        stateGateAngles(2 * M) -= xacc::constants::pi / 4.0;
        kernel = statePreparationCircuit(stateGateAngles);
        kernel->addVariables(entangler->getVariables());
        for (auto &inst : entangler->getInstructions()) {
          kernel->addInstruction(inst);
        }
        first -= vqeWrapper(observable, kernel, x);
      }

      first /= 2.0;

      auto second = jacobian(M, state, rotatedEigenstates.col(state));
      stateGrad(M) += first * second * rotatedEigenstates(M, state);
    }
    gradients.push_back(stateGrad);
  }

  int nParams = 1;
  std::vector<double> tmp_x;
  // RHS #2 Eq. 134
  for (int state = 0; state < nStates; state++) {

    double grad = 0.0;

    for (int g = 0; g < nParams; g++) {

      for (int M = 0; M < nChromophores; M++) {

        double first = 0.0;
        Eigen::VectorXd stateGateAngles;

        if (M == 0) {

          tmp_x = x;
          tmp_x[g] += xacc::constants::pi / 4.0;

          stateGateAngles = CISGateAngles.col(state);
          stateGateAngles(M) += xacc::constants::pi / 4.0;
          kernel = statePreparationCircuit(stateGateAngles);
          kernel->addVariables(entangler->getVariables());
          for (auto &inst : entangler->getInstructions()) {
            kernel->addInstruction(inst);
          }
          first += vqeWrapper(observable, kernel, tmp_x);

          stateGateAngles = CISGateAngles.col(state);
          stateGateAngles(M) -= xacc::constants::pi / 4.0;
          kernel = statePreparationCircuit(stateGateAngles);
          kernel->addVariables(entangler->getVariables());
          for (auto &inst : entangler->getInstructions()) {
            kernel->addInstruction(inst);
          }
          first -= vqeWrapper(observable, kernel, tmp_x);

          tmp_x = x;
          tmp_x[g] -= xacc::constants::pi / 4.0;
          stateGateAngles = CISGateAngles.col(state);
          stateGateAngles(M) += xacc::constants::pi / 4.0;
          kernel = statePreparationCircuit(stateGateAngles);
          kernel->addVariables(entangler->getVariables());
          for (auto &inst : entangler->getInstructions()) {
            kernel->addInstruction(inst);
          }
          first -= vqeWrapper(observable, kernel, tmp_x);

          stateGateAngles = CISGateAngles.col(state);
          stateGateAngles(M) -= xacc::constants::pi / 4.0;
          kernel = statePreparationCircuit(stateGateAngles);
          kernel->addVariables(entangler->getVariables());
          for (auto &inst : entangler->getInstructions()) {
            kernel->addInstruction(inst);
          }
          first += vqeWrapper(observable, kernel, tmp_x);

        } else {
          // Eq. 133

          tmp_x = x;
          tmp_x[g] += xacc::constants::pi / 4.0;

          stateGateAngles = CISGateAngles.col(state);
          stateGateAngles(2 * M - 1) += xacc::constants::pi / 4.0;
          kernel = statePreparationCircuit(stateGateAngles);
          kernel->addVariables(entangler->getVariables());
          for (auto &inst : entangler->getInstructions()) {
            kernel->addInstruction(inst);
          }
          first -= vqeWrapper(observable, kernel, tmp_x);

          stateGateAngles = CISGateAngles.col(state);
          stateGateAngles(2 * M - 1) -= xacc::constants::pi / 4.0;
          kernel = statePreparationCircuit(stateGateAngles);
          kernel->addVariables(entangler->getVariables());
          for (auto &inst : entangler->getInstructions()) {
            kernel->addInstruction(inst);
          }
          first += vqeWrapper(observable, kernel, tmp_x);

          stateGateAngles = CISGateAngles.col(state);
          stateGateAngles(2 * M) += xacc::constants::pi / 4.0;
          kernel = statePreparationCircuit(stateGateAngles);
          kernel->addVariables(entangler->getVariables());
          for (auto &inst : entangler->getInstructions()) {
            kernel->addInstruction(inst);
          }
          first += vqeWrapper(observable, kernel, tmp_x);

          stateGateAngles = CISGateAngles.col(state);
          stateGateAngles(2 * M) -= xacc::constants::pi / 4.0;
          kernel = statePreparationCircuit(stateGateAngles);
          kernel->addVariables(entangler->getVariables());
          for (auto &inst : entangler->getInstructions()) {
            kernel->addInstruction(inst);
          }
          first -= vqeWrapper(observable, kernel, tmp_x);

          tmp_x = x;
          tmp_x[g] -= xacc::constants::pi / 4.0;

          stateGateAngles = CISGateAngles.col(state);
          stateGateAngles(2 * M - 1) += xacc::constants::pi / 4.0;
          kernel = statePreparationCircuit(stateGateAngles);
          kernel->addVariables(entangler->getVariables());
          for (auto &inst : entangler->getInstructions()) {
            kernel->addInstruction(inst);
          }
          first += vqeWrapper(observable, kernel, tmp_x);

          stateGateAngles = CISGateAngles.col(state);
          stateGateAngles(2 * M - 1) -= xacc::constants::pi / 4.0;
          kernel = statePreparationCircuit(stateGateAngles);
          kernel->addVariables(entangler->getVariables());
          for (auto &inst : entangler->getInstructions()) {
            kernel->addInstruction(inst);
          }
          first -= vqeWrapper(observable, kernel, tmp_x);

          stateGateAngles = CISGateAngles.col(state);
          stateGateAngles(2 * M) += xacc::constants::pi / 4.0;
          kernel = statePreparationCircuit(stateGateAngles);
          kernel->addVariables(entangler->getVariables());
          for (auto &inst : entangler->getInstructions()) {
            kernel->addInstruction(inst);
          }
          first -= vqeWrapper(observable, kernel, tmp_x);

          stateGateAngles = CISGateAngles.col(state);
          stateGateAngles(2 * M) -= xacc::constants::pi / 4.0;
          kernel = statePreparationCircuit(stateGateAngles);
          kernel->addVariables(entangler->getVariables());
          for (auto &inst : entangler->getInstructions()) {
            kernel->addInstruction(inst);
          }
          first += vqeWrapper(observable, kernel, tmp_x);
        }

        auto second = jacobian(M, state, CISEigenstates.col(state));
        grad += vqeMultipliers[state](g) * first * second / nStates;
      }
    }
  }

  // compute Hessian Eq. 135
  std::vector<Eigen::MatrixXd> hessian;
  for (int state = 0; state < nStates; state++) {

    Eigen::MatrixXd stateHessian = Eigen::MatrixXd::Zero(nStates, nStates);
    for (int I = 0; I < nStates; I++) {
      for (int Ip = I; Ip < nStates; Ip++) {

        stateHessian(I, Ip) = CISMatrix(I, Ip) - 2.0 * CISEnergies(state) * CISEigenstates(I, state) * CISEigenstates(Ip, state);

        if( I == Ip) {
          stateHessian(I, Ip) -= CISEnergies(state);
        }
      }
    }
  }

  // solve systems Eq. 136
  std::vector<Eigen::VectorXd> multipliers;
  for (int state = 0; state < nStates; state++) {
    multipliers.push_back(hessian[state].colPivHouseholderQr().solve(-gradients[state]));
  }

  // Compute D Eq. 138
  std::vector<Eigen::MatrixXd> D;
  for (int state = 0; state < nStates; state++) {
    for (int I = 0; I < nStates; I++) {
      for (int Ip = I; Ip < nStates; Ip++) {



      }
    }
    
  }

  std::vector<Eigen::VectorXd> CRS1PDM;

  return CRS1PDM;
}
*/

} // namespace algorithm
} // namespace xacc
