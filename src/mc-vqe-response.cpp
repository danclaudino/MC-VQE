#include "Monomer.hpp"
#include "PauliOperator.hpp"
#include "mc-vqe.hpp"
#include "xacc.hpp"
#include <map>
#include <memory>
#include <vector>

using namespace xacc;
using namespace xacc::quantum;

namespace xacc {
namespace algorithm {

// compute the unrelaxed 1PDM
// Eq. 104/121
std::map<std::string, std::vector<Eigen::MatrixXd>>
MC_VQE::getUnrelaxed1PDM(const std::vector<double> &x) {

  Eigen::MatrixXd rotatedEigenstates = CISEigenstates * subSpaceRotation;
  Eigen::MatrixXd gateAngles = statePreparationAngles(rotatedEigenstates);
  std::map<std::string, std::vector<Eigen::MatrixXd>> unrelaxed1PDM;

  // loop over the single Pauli term
  for (auto termStr : {"X", "Z"}) {

    // each element of term1PDM is a vector of density matrix elements
    // for each state given the termStr Pauli term
    std::vector<Eigen::MatrixXd> term1PDM;
    // loop over the MC states
    for (int state = 0; state < nStates; state++) {

      // prepare interference state and append entangler
      auto kernel = statePreparationCircuit(gateAngles.col(state));
      kernel->addVariables(entangler->getVariables());
      kernel->addInstructions(entangler->getInstructions());

      // loop over chromophores and compute <term>
      // stateDensityMatrix stores 1PDM elements for each chromophore in state
      Eigen::MatrixXd stateDensityMatrix = Eigen::VectorXd::Zero(nChromophores);
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
std::map<std::string, std::vector<Eigen::MatrixXd>>
MC_VQE::getUnrelaxed2PDM(const std::vector<double> &x) {

  Eigen::MatrixXd rotatedEigenstates = CISEigenstates * subSpaceRotation;
  Eigen::MatrixXd gateAngles = statePreparationAngles(rotatedEigenstates);
  std::map<std::string, std::vector<Eigen::MatrixXd>> unrelaxed2PDM;

  // loop over the Pauli products
  for (auto termStr : {"XX", "XZ", "ZZ"}) {

    // each element of term2PDM is a vector of density matrix elements
    // for each state given the termStr Pauli term
    std::vector<Eigen::MatrixXd> term2PDM;
    for (int state = 0; state < nStates; state++) {

      // prepare interference state and append entangler
      auto kernel = statePreparationCircuit(gateAngles.col(state));
      kernel->addVariables(entangler->getVariables());
      kernel->addInstructions(entangler->getInstructions());

      // loop over chromophores and compute <term>
      // stateDensityMatrix stores 1PDM elements for each chromophore in state
      Eigen::MatrixXd stateDensityMatrix =
          Eigen::MatrixXd::Zero(nChromophores, nChromophores);
      for (int A = 0; A < nChromophores; A++) {
        for (int B : pairs[A]) {

          if (((termStr == "XX") || (termStr == "ZZ")) && (B < A)) {
            continue;
          }
          std::string term1 = {termStr[0]}, term2 = {termStr[1]};
          auto term = std::make_shared<PauliOperator>(
              PauliOperator({{A, term1}, {B, term2}}));
          stateDensityMatrix(A, B) = vqeWrapper(term, kernel, x);
          if ((termStr == "XX") || (termStr == "ZZ")) {
            stateDensityMatrix(B, A) = stateDensityMatrix(A, B);
          }
        }
      }
      term2PDM.push_back(stateDensityMatrix);
    }
    unrelaxed2PDM.emplace(termStr, term2PDM);

    if (termStr == "XZ") {
      std::vector<Eigen::MatrixXd> term2PDM;
      for (auto &dm : unrelaxed2PDM["XZ"]) {
        term2PDM.push_back(dm.transpose());
      }
      unrelaxed2PDM.emplace("ZX", term2PDM);
    }
  }

  return unrelaxed2PDM;
}

// compute the VQE multipliers
std::vector<Eigen::MatrixXd>
MC_VQE::getVQEMultipliers(const std::vector<double> &x) {

  Eigen::MatrixXd rotatedEigenstates = CISEigenstates * subSpaceRotation;
  Eigen::MatrixXd gateAngles = statePreparationAngles(rotatedEigenstates);
  auto nParams = x.size();
  std::vector<double> tmp_x;

  std::vector<Eigen::MatrixXd> gradients;
  // compute gradient for each state energy w.r.t. to entangler parameters
  // Eq. 124
  for (int state = 0; state < nStates; state++) {

    // prepare interference state and append entangler
    auto kernel = statePreparationCircuit(gateAngles.col(state));
    kernel->addVariables(entangler->getVariables());
    kernel->addInstructions(entangler->getInstructions());

    Eigen::MatrixXd stateMultipliers = Eigen::VectorXd::Zero(nParams);
    for (int g = 0; g < nParams; g++) {
      tmp_x = x;
      tmp_x[g] += PI / 2.0;
      stateMultipliers(g) = vqeWrapper(observable, kernel, tmp_x);

      tmp_x = x;
      tmp_x[g] -= PI / 2.0;
      stateMultipliers(g) -= vqeWrapper(observable, kernel, tmp_x);
    }
    gradients.push_back(stateMultipliers);
  }

  // Now compute Hessian
  Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(nParams, nParams);
  for (int state = 0; state < nStates; state++) {

    // prepare interference state and append entangler
    auto kernel = statePreparationCircuit(CISGateAngles.col(state));
    kernel->addVariables(entangler->getVariables());
    kernel->addInstructions(entangler->getInstructions());

    // diagonal Hessian elements
    Eigen::MatrixXd stateHessian = Eigen::MatrixXd::Zero(nParams, nParams);
    for (int g = 0; g < nParams; g++) {

      // Eq. 125 term #1
      tmp_x = x;
      tmp_x[g] += PI;
      auto p = vqeWrapper(observable, kernel, tmp_x);

      // Eq. 125 term #2
      auto z = -2.0 * vqeWrapper(observable, kernel, x);

      // Eq. 125 term #3
      tmp_x = x;
      tmp_x[g] -= PI;
      auto m = vqeWrapper(observable, kernel, tmp_x);

      stateHessian(g, g) = p + z + m;
    }

    // off-diagonal Hessian elements
    for (int g = 0; g < nParams; g++) {
      for (int gp = 0; gp < g; gp++) {

        // Eq. 126 term #1
        tmp_x = x;
        tmp_x[g] += PI / 2.0;
        tmp_x[gp] += PI / 2.0;
        auto pp = vqeWrapper(observable, kernel, tmp_x);

        // Eq. 126 term #2
        tmp_x = x;
        tmp_x[g] += PI / 2.0;
        tmp_x[gp] -= PI / 2.0;
        auto pm = vqeWrapper(observable, kernel, tmp_x);

        // Eq. 126 term #3
        tmp_x = x;
        tmp_x[g] -= PI / 2.0;
        tmp_x[gp] += PI / 2.0;
        auto mp = vqeWrapper(observable, kernel, tmp_x);

        // Eq. 126 term #4
        tmp_x = x;
        tmp_x[g] -= PI / 2.0;
        tmp_x[gp] -= PI / 2.0;
        auto mm = vqeWrapper(observable, kernel, tmp_x);

        stateHessian(g, gp) = stateHessian(gp, g) = pp - pm - mp + mm;
      }
    }
    hessian += stateHessian;
  }
  hessian /= nStates;

  // Eq. 127
  std::vector<Eigen::MatrixXd> vqeMultipliers;
  for (int state = 0; state < nStates; state++) {
    vqeMultipliers.push_back(
        hessian.colPivHouseholderQr().solve(-gradients[state]));
  }

  return vqeMultipliers;
}

// compute CP-SA-VQE 1PDM
std::map<std::string, std::vector<Eigen::MatrixXd>>
MC_VQE::getVQE1PDM(const std::vector<double> &x,
                   const std::vector<Eigen::MatrixXd> &vqeMultipliers) {

  // Eq. 128
  auto nParams = x.size();
  std::vector<double> tmp_x;
  std::map<std::string, std::vector<Eigen::MatrixXd>> vqe1PDM;
  for (auto termStr : {"X", "Z"}) {

    std::vector<Eigen::MatrixXd> stateAveraged1PDM;
    for (int mcState = 0; mcState < nStates; mcState++) {

      Eigen::MatrixXd term1PDM = Eigen::VectorXd::Zero(nChromophores);
      // sum over all states
      for (int state = 0; state < nStates; state++) {

        // prepare interference state and append entangler
        auto kernel = statePreparationCircuit(CISGateAngles.col(state));
        kernel->addVariables(entangler->getVariables());
        kernel->addInstructions(entangler->getInstructions());
        for (int A = 0; A < nChromophores; A++) {

          auto term =
              std::make_shared<PauliOperator>(PauliOperator({{A, termStr}}));
          // sum over all VQE parameters
          for (int g = 0; g < nParams; g++) {

            tmp_x = x;
            tmp_x[g] += PI / 2.0;
            term1PDM(A) +=
                vqeMultipliers[mcState](g) * vqeWrapper(term, kernel, tmp_x);

            tmp_x = x;
            tmp_x[g] -= PI / 2.0;
            term1PDM(A) -=
                vqeMultipliers[mcState](g) * vqeWrapper(term, kernel, tmp_x);
          }
        }
      }
      stateAveraged1PDM.push_back(term1PDM / nStates);
    }
    vqe1PDM.emplace(termStr, stateAveraged1PDM);
  }

  return vqe1PDM;
}

// compute CP-SA-VQE 2PDM
std::map<std::string, std::vector<Eigen::MatrixXd>>
MC_VQE::getVQE2PDM(const std::vector<double> &x,
                   const std::vector<Eigen::MatrixXd> &vqeMultipliers) {

  // Eq. 128
  auto nParams = x.size();
  std::vector<double> tmp_x;
  std::map<std::string, std::vector<Eigen::MatrixXd>> vqe2PDM;
  for (auto termStr : {"XX", "XZ", "ZZ"}) {

    std::vector<Eigen::MatrixXd> stateAveraged2PDM;
    for (int mcState = 0; mcState < nStates; mcState++) {

      Eigen::MatrixXd term2PDM =
          Eigen::MatrixXd::Zero(nChromophores, nChromophores);
      // sum over all states
      for (int state = 0; state < nStates; state++) {

        // prepare interference state and append entangler
        auto kernel = statePreparationCircuit(CISGateAngles.col(state));
        kernel->addVariables(entangler->getVariables());
        kernel->addInstructions(entangler->getInstructions());
        for (int A = 0; A < nChromophores; A++) {

          for (auto B : pairs[A]) {

            if (((termStr == "XX") || (termStr == "ZZ")) && (B < A)) {
              continue;
            }
            std::string term1 = {termStr[0]}, term2 = {termStr[1]};
            auto term = std::make_shared<PauliOperator>(
                PauliOperator({{A, term1}, {B, term2}}));
            // sum over all VQE parameters
            for (int g = 0; g < nParams; g++) {

              tmp_x = x;
              tmp_x[g] += PI / 2.0;
              term2PDM(A, B) +=
                  vqeMultipliers[mcState](g) * vqeWrapper(term, kernel, tmp_x);

              tmp_x = x;
              tmp_x[g] -= PI / 2.0;
              term2PDM(A, B) -=
                  vqeMultipliers[mcState](g) * vqeWrapper(term, kernel, tmp_x);
            }

            if ((termStr == "XX") || (termStr == "ZZ")) {
              term2PDM(B, A) = term2PDM(A, B);
            }
          }
        }
      }
      stateAveraged2PDM.push_back(term2PDM / nStates);
    }
    vqe2PDM.emplace(termStr, stateAveraged2PDM);

    if (termStr == "XZ") {
      std::vector<Eigen::MatrixXd> term2PDM;
      for (auto &dm : vqe2PDM["XZ"]) {
        term2PDM.push_back(dm.transpose());
      }
      vqe2PDM.emplace("ZX", term2PDM);
    }
  }

  return vqe2PDM;
}

Eigen::MatrixXd
MC_VQE::getStatePrepationAngleGradient(const Eigen::MatrixXd &gateAngles,
                                       const std::vector<double> &x) {

  Eigen::MatrixXd stateGateAngles = Eigen::VectorXd::Zero(gateAngles.size());
  Eigen::MatrixXd stateAngleGrad = Eigen::VectorXd::Zero(gateAngles.size() - 1);
  std::shared_ptr<CompositeInstruction> kernel;

  for (int M = 0; M < gateAngles.size() - 1; M++) {

    double angleGrad = 0.0;
    // Eq. 132
    if (M == 0) {

      stateGateAngles = gateAngles;
      stateGateAngles(M) += PI / 2.0;
      kernel = statePreparationCircuit(stateGateAngles);
      kernel->addVariables(entangler->getVariables());
      kernel->addInstructions(entangler->getInstructions());
      angleGrad -= vqeWrapper(observable, kernel, x);

      stateGateAngles = gateAngles;
      stateGateAngles(M) -= PI / 2.0;
      kernel = statePreparationCircuit(stateGateAngles);
      kernel->addVariables(entangler->getVariables());
      kernel->addInstructions(entangler->getInstructions());
      angleGrad += vqeWrapper(observable, kernel, x);

      // Eq. 133
    } else {

      stateGateAngles = gateAngles;
      stateGateAngles(2 * M - 1) += PI;
      kernel = statePreparationCircuit(stateGateAngles);
      kernel->addVariables(entangler->getVariables());
      kernel->addInstructions(entangler->getInstructions());
      angleGrad += vqeWrapper(observable, kernel, x);

      stateGateAngles = gateAngles;
      stateGateAngles(2 * M - 1) -= PI;
      kernel = statePreparationCircuit(stateGateAngles);
      kernel->addVariables(entangler->getVariables());
      kernel->addInstructions(entangler->getInstructions());
      angleGrad -= vqeWrapper(observable, kernel, x);

      stateGateAngles = gateAngles;
      stateGateAngles(2 * M) += PI;
      kernel = statePreparationCircuit(stateGateAngles);
      kernel->addVariables(entangler->getVariables());
      kernel->addInstructions(entangler->getInstructions());
      angleGrad += vqeWrapper(observable, kernel, x);

      stateGateAngles = gateAngles;
      stateGateAngles(2 * M) -= PI;
      kernel = statePreparationCircuit(stateGateAngles);
      kernel->addVariables(entangler->getVariables());
      kernel->addInstructions(entangler->getInstructions());
      angleGrad -= vqeWrapper(observable, kernel, x);
      angleGrad /= 2.0;
    }

    stateAngleGrad(M) = angleGrad;
  }

  return stateAngleGrad;
}

// compute multipliers w.r.t. contracted reference states
std::vector<Eigen::MatrixXd>
MC_VQE::getCRSMultipliers(const std::vector<double> &x,
                          const std::vector<Eigen::MatrixXd> &vqeMultipliers) {

  // Jacobian Eq. 62
  auto jacobian = [](int M, int I, Eigen::VectorXd coefficients) {
    auto size = coefficients.size();
    double ret = 0.0;

    if (M == I) {
      ret -= 1.0 / coefficients.tail(size - M - 1).norm();
    }

    if (I >= M) {
      ret += coefficients(M) * coefficients(I) /
             (coefficients.tail(size - M - 1).norm() *
              coefficients.tail(size - M).squaredNorm());
    }

    if ((M == size - 1) && (coefficients(Eigen::last) < 0.0)) {
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
  // compute angle gradients
  Eigen::MatrixXd stateAngleGrad = Eigen::VectorXd::Zero(gateAngles.rows() - 1);
  for (int state = 0; state < nStates; state++) {

    stateAngleGrad = getStatePrepationAngleGradient(gateAngles.col(state), x);

    // TODO
    // if (state == 1) stateAngleGrad(0) *= -1.0;

    Eigen::MatrixXd tmp = Eigen::VectorXd::Zero(gateAngles.rows());
    for (int M = 0; M < gateAngles.rows() - 1; M++) {
      for (int I = 0; I < nStates + 1; I++) {

        tmp(I) +=
            stateAngleGrad(M) * jacobian(M, I, rotatedEigenstates.col(state));
      }
    }

    Eigen::MatrixXd stateGrad =
        Eigen::MatrixXd::Zero(gateAngles.rows(), nStates);
    for (int M = 0; M < gateAngles.rows() - 1; M++) {
      for (int I = 0; I < nStates + 1; I++) {
        stateGrad(I, M) = tmp(I) * subSpaceRotation(state, M);
      }
    }

    gradients.push_back(stateGrad);
  }

  Eigen::MatrixXd cisGradient =
      Eigen::MatrixXd::Zero(gateAngles.rows(), nStates);

  for (int state = 0; state < nStates; state++) {

    Eigen::MatrixXd tmp = Eigen::VectorXd::Zero(gateAngles.rows());

    for (int g = 0; g < gateAngles.rows() - 1; g++) {

      auto plus_x = x;
      plus_x[g] += PI / 2;
      Eigen::VectorXd plus_G =
          getStatePrepationAngleGradient(CISGateAngles.col(state), plus_x);

      auto minus_x = x;
      minus_x[g] -= PI / 2;
      Eigen::VectorXd minus_G =
          getStatePrepationAngleGradient(CISGateAngles.col(state), minus_x);

      Eigen::VectorXd stateAngleGrad = plus_G - minus_G;

      // TODO: I can't figure out why this sign flips compared to Rob's code
      // will leave this here for now to check the rest of the code
      // if (state ==1) stateAngleGrad(0) *= -1.0;

      for (int M = 0; M < gateAngles.rows() - 1; M++) {
        for (int I = 0; I < nStates + 1; I++) {

          tmp(I) += stateAngleGrad(M) *
                    jacobian(M, I, CISEigenstates.col(state)) *
                    vqeMultipliers[state](g);
        }
      }
    }
    cisGradient.col(state) = tmp / nStates;
  }

  for (int state = 0; state < nStates; state++) {
    gradients[state] += cisGradient;
  }

  // compute Hessian Eq. 135
  std::vector<Eigen::MatrixXd> hessian;
  for (int state = 0; state < nStates; state++) {

    Eigen::MatrixXd stateHessian = CISHamiltonian;
    stateHessian -= CISEnergies(state) *
                    Eigen::MatrixXd::Identity(nStates + 1, nStates + 1);
    stateHessian -= 2.0 * CISEnergies(state) * CISEigenstates.col(state) *
                    CISEigenstates.col(state).transpose();
    hessian.push_back(stateHessian);
  }

  // Eigen::MatrixXd G1 = Eigen::MatrixXd::Zero(nStates + 1, nStates);
  // Eigen::MatrixXd G2 = Eigen::MatrixXd::Zero(nStates + 1, nStates);
  // G1 << 0.00110366, 0.00269595, 0.00589268, -0.00094224, 0.00400579,
  //    -0.00209204;
  // G2 << 0.00061209, -0.0017159, -0.00178271, 0.00328989, 0.00312522,
  // 0.00616533; gradients.clear(); gradients.push_back(-G1);
  // gradients.push_back(-G2);

  // Eq. 136
  std::vector<Eigen::MatrixXd> cpCRSMultipliers;
  for (int vqeState = 0; vqeState < nStates; vqeState++) {
    Eigen::MatrixXd stateMultipliers =
        Eigen::MatrixXd::Zero(nStates + 1, nStates);
    for (int cisState = 0; cisState < nStates; cisState++) {
      stateMultipliers.col(cisState) =
          hessian[cisState].colPivHouseholderQr().solve(
              -gradients[vqeState].col(cisState));
    }
    cpCRSMultipliers.push_back(stateMultipliers);
  }

  return cpCRSMultipliers;
}

// compute CRS contribution to the relaxed DM
// because we don't need to getObservable from string
// we can have a single call for 1 and 2 PDMs
std::map<std::string, std::vector<Eigen::MatrixXd>>
MC_VQE::getCRSDensityMatrices(
    const std::vector<Eigen::MatrixXd> &cpCRSMultipliers) {

  std::map<std::string, std::vector<Eigen::MatrixXd>> crsPDM;
  for (int mcState = 0; mcState < nStates; mcState++) {
    // Compute D Eq. 138

    Eigen::MatrixXd D = cpCRSMultipliers[mcState] * CISEigenstates.transpose();
    for (int cisState = 0; cisState < nStates; cisState++) {

      double sum = cpCRSMultipliers[mcState].col(cisState).transpose() *
                   CISEigenstates.col(cisState);
      D -= sum * CISEigenstates.col(cisState) *
           CISEigenstates.col(cisState).transpose();
    }

    // D << -0.01178758, -0.01051503, -0.02515445, -0.01051503, 0.01726251,
    //    0.0133979, -0.02515445, 0.0133979, -0.00547493;

    std::vector<Eigen::MatrixXd> termPDM;
    for (auto termStr : {"X", "Z", "XX", "XZ", "ZZ"}) {
      crsPDM.emplace(termStr, termPDM);
      if (termStr == "XZ") {
        crsPDM.emplace("ZX", termPDM);
      }

      Eigen::MatrixXd stateDensityMatrix;

      if (termStr == "X" || termStr == "Z") {
        stateDensityMatrix = Eigen::VectorXd::Zero(nChromophores);
      } else {
        stateDensityMatrix =
            Eigen::MatrixXd::Zero(nChromophores, nChromophores);
      }

      for (int A = 0; A < nChromophores; A++) {

        // sum of all derivatives of Eq. 50
        // H is symmetric
        if (termStr == "X") {
          // CIS Hamiltonian gradient
          Eigen::MatrixXd cisHamiltonianGrad =
              Eigen::MatrixXd::Zero(nChromophores + 1, nChromophores + 1);
          cisHamiltonianGrad(0, A + 1) = cisHamiltonianGrad(A + 1, 0) = 1.0;
          // dot D with cisHamiltonianGrad
          stateDensityMatrix(A) =
              (D.transpose() * cisHamiltonianGrad).diagonal().sum();
        }

        if (termStr == "Z") {
          // CIS Hamiltonian gradient
          Eigen::MatrixXd cisHamiltonianGrad =
              Eigen::MatrixXd::Identity(nChromophores + 1, nChromophores + 1);
          cisHamiltonianGrad(A + 1, A + 1) -= 2.0;
          // dot D with cisHamiltonianGrad
          stateDensityMatrix(A) =
              (D.transpose() * cisHamiltonianGrad).diagonal().sum();
        }

        // not sure why, but my 2PDMs come out as half of Rob's
        if (termStr == "ZZ") {
          // CIS Hamiltonian gradient
          Eigen::MatrixXd cisHamiltonianGrad =
              Eigen::MatrixXd::Identity(nChromophores + 1, nChromophores + 1);
          cisHamiltonianGrad *= 0.5;
          cisHamiltonianGrad(A + 1, A + 1) -= 1.0;
          // dot D with cisHamiltonianGrad
          double value = (D.transpose() * cisHamiltonianGrad).diagonal().sum();
          for (auto B : pairs[A]) {
            stateDensityMatrix(A, B) += 2.0 * value;
            stateDensityMatrix(B, A) += 2.0 * value;
          }
        }

        if (termStr == "XZ") {
          // CIS Hamiltonian gradient
          Eigen::MatrixXd cisHamiltonianGrad =
              Eigen::MatrixXd::Zero(nChromophores + 1, nChromophores + 1);
          cisHamiltonianGrad(A + 1, 0) = 0.5;
          cisHamiltonianGrad(0, A + 1) = 0.5;
          // dot D with cisHamiltonianGrad
          double value = (D.transpose() * cisHamiltonianGrad).diagonal().sum();
          for (auto B : pairs[A]) {
            stateDensityMatrix(A, B) += 2.0 * value;
          }
        }

        if (termStr == "XX") {
          // CIS Hamiltonian gradient
          Eigen::MatrixXd cisHamiltonianGrad =
              Eigen::MatrixXd::Zero(nChromophores + 1, nChromophores + 1);
          for (auto B : pairs[A]) {
            cisHamiltonianGrad(A + 1, B + 1) = 1.0;
            double value =
                (D.transpose() * cisHamiltonianGrad).diagonal().sum();
            // dot D with cisHamiltonianGrad
            stateDensityMatrix(A, B) += 2.0 * value;
          }
        }
      }

      crsPDM[termStr].push_back(stateDensityMatrix);
      if (termStr == "XZ") {
        crsPDM["ZX"].push_back(stateDensityMatrix.transpose());
      }
    }
  }

  return crsPDM;
}

std::map<std::string, std::vector<Eigen::MatrixXd>>
MC_VQE::getRelaxedDensityMatrices(
    std::map<std::string, std::vector<Eigen::MatrixXd>> &unrelaxedDM,
    std::map<std::string, std::vector<Eigen::MatrixXd>> &vqeDM,
    std::map<std::string, std::vector<Eigen::MatrixXd>> &crsDM) {

  std::map<std::string, std::vector<Eigen::MatrixXd>> relaxedDM;

  for (auto termStr : {"X", "Z", "XX", "XZ", "ZX", "ZZ"}) {

    std::vector<Eigen::MatrixXd> DMs(nStates);
    relaxedDM.emplace(termStr, DMs);
    for (int state = 0; state < nStates; state++) {

      relaxedDM[termStr][state] = unrelaxedDM[termStr][state] +
                                  vqeDM[termStr][state] + crsDM[termStr][state];
    }
  }

  return relaxedDM;
}

std::map<std::string, std::vector<Eigen::MatrixXd>>
MC_VQE::getMonomerBasisDensityMatrices(
    std::map<std::string, std::vector<Eigen::MatrixXd>> &_DM) {

  // the 1PDM is straightforward, so I only do this for the 2PDM
  // here we change from Pauli to monomer basis
  std::map<std::string, std::vector<Eigen::MatrixXd>> monomerBasisDM;
  for (auto termStr : {"HH", "HT", "HP", "TH", "TT", "TP", "PH", "PT", "PP"}) {

    std::vector<Eigen::MatrixXd> termDM;
    for (int state = 0; state < nStates; state++) {

      Eigen::MatrixXd stateDensityMatrix =
          Eigen::MatrixXd::Zero(nChromophores, nChromophores);

      // these are transpose of matrices previously computed
      if (termStr == "TH") {
        stateDensityMatrix = monomerBasisDM["HT"][state].transpose();
      } else if (termStr == "PH") {
        stateDensityMatrix = monomerBasisDM["HP"][state].transpose();
      } else if (termStr == "PT") {
        stateDensityMatrix = monomerBasisDM["TP"][state].transpose();
      } else {

        for (int A = 0; A < nChromophores; A++) {
          for (auto B : pairs[A]) {

            if (termStr == "HH") {

              stateDensityMatrix(A, B) += 1.0 / 4.0;
              stateDensityMatrix(A, B) += _DM["Z"][state](A) / 4.0;
              stateDensityMatrix(A, B) += _DM["Z"][state](B) / 4.0;
              stateDensityMatrix(A, B) += _DM["ZZ"][state](A, B) / 4.0;
            }

            if (termStr == "HT") {
              stateDensityMatrix(A, B) += _DM["X"][state](B) / 2.0;
              stateDensityMatrix(A, B) += _DM["ZX"][state](A, B) / 2.0;
            }

            if (termStr == "HP") {
              stateDensityMatrix(A, B) += 1.0 / 4.0;
              stateDensityMatrix(A, B) += _DM["Z"][state](A) / 4.0;
              stateDensityMatrix(A, B) -= _DM["Z"][state](B) / 4.0;
              stateDensityMatrix(A, B) -= _DM["ZZ"][state](A, B) / 4.0;
            }

            if (termStr == "TT") {
              stateDensityMatrix(A, B) += _DM["XX"][state](A, B);
            }

            if (termStr == "TP") {
              stateDensityMatrix(A, B) += _DM["X"][state](A) / 2.0;
              stateDensityMatrix(A, B) -= _DM["XZ"][state](A, B) / 2.0;
            }

            if (termStr == "PP") {
              stateDensityMatrix(A, B) += 1.0 / 4.0;
              stateDensityMatrix(A, B) -= _DM["Z"][state](A) / 4.0;
              stateDensityMatrix(A, B) -= _DM["Z"][state](B) / 4.0;
              stateDensityMatrix(A, B) += _DM["ZZ"][state](A, B) / 4.0;
            }
          }
        }
      }

      termDM.push_back(stateDensityMatrix);
    }
    monomerBasisDM.emplace(termStr, termDM);
  }

  return monomerBasisDM;
}

std::map<std::string, std::vector<Eigen::MatrixXd>> MC_VQE::getMonomerGradient(
    std::map<std::string, std::vector<Eigen::MatrixXd>> &_1PDM) {

  std::map<std::string, std::vector<Eigen::MatrixXd>> monomerGradients;

  // Eq. 143
  for (auto termStr : {"EH", "EP", "ET"}) {

    std::vector<Eigen::MatrixXd> termGrad;

    for (int state = 0; state < nStates; state++) {

      Eigen::MatrixXd grad = Eigen::VectorXd::Constant(nChromophores, 0.5);

      if (termStr == "EH") {
        grad += 0.5 * _1PDM["Z"][state];
      }

      if (termStr == "EP") {
        grad -= 0.5 * _1PDM["Z"][state];
      }

      termGrad.push_back(grad);
    }
    monomerGradients.emplace(termStr, termGrad);
  }

  return monomerGradients;
}

std::map<std::string, std::vector<Eigen::MatrixXd>>
MC_VQE::getDimerInteractionGradient(
    std::map<std::string, std::vector<Eigen::MatrixXd>> &_2PDM) {

  // Eq. 147
  auto dipolePartial = [&](const Eigen::Vector3d mu, const Eigen::Vector3d r) {
    double d = r.norm();
    Eigen::Vector3d ret = Eigen::Vector3d::Zero();
    ret = mu / std::pow(d, 3.0) - 3.0 * mu.dot(r) * r / std::pow(d, 5.0);
    return ret;
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

  std::map<std::string, std::vector<Eigen::MatrixXd>> dimerInteractionGradients;

  for (auto termStr : {"MH", "MT", "MP", "R"}) {

    std::vector<Eigen::MatrixXd> termEta;
    for (int state = 0; state < nStates; state++) {

      Eigen::MatrixXd eta = Eigen::MatrixXd::Zero(nChromophores, 3);
      for (int A = 0; A < nChromophores; A++) {

        for (auto B : pairs[A]) {

          auto r =
              monomers[B].getCenterOfMass() - monomers[A].getCenterOfMass();

          // Eq. 145
          if (termStr == "MH") {
            eta.row(A) +=
                dipolePartial(monomers[B].getGroundStateDipole(), r) *
                    _2PDM["HH"][state](A, B) +
                dipolePartial(monomers[B].getTransitionDipole(), r) *
                    _2PDM["HT"][state](A, B) +
                dipolePartial(monomers[B].getExcitedStateDipole(), r) *
                    _2PDM["HP"][state](A, B);
            eta.row(B) +=
                dipolePartial(monomers[A].getGroundStateDipole(), r) *
                    _2PDM["HH"][state](A, B) +
                dipolePartial(monomers[A].getTransitionDipole(), r) *
                    _2PDM["TH"][state](A, B) +
                dipolePartial(monomers[A].getExcitedStateDipole(), r) *
                    _2PDM["PH"][state](A, B);
          }

          if (termStr == "MT") {
            eta.row(A) +=
                dipolePartial(monomers[B].getGroundStateDipole(), r) *
                    _2PDM["TH"][state](A, B) +
                dipolePartial(monomers[B].getTransitionDipole(), r) *
                    _2PDM["TT"][state](A, B) +
                dipolePartial(monomers[B].getExcitedStateDipole(), r) *
                    _2PDM["TP"][state](A, B);
            eta.row(B) +=
                dipolePartial(monomers[A].getGroundStateDipole(), r) *
                    _2PDM["HT"][state](A, B) +
                dipolePartial(monomers[A].getTransitionDipole(), r) *
                    _2PDM["TT"][state](A, B) +
                dipolePartial(monomers[A].getExcitedStateDipole(), r) *
                    _2PDM["PT"][state](A, B);
          }

          if (termStr == "MP") {
            eta.row(A) +=
                dipolePartial(monomers[B].getGroundStateDipole(), r) *
                    _2PDM["PH"][state](A, B) +
                dipolePartial(monomers[B].getTransitionDipole(), r) *
                    _2PDM["PT"][state](A, B) +
                dipolePartial(monomers[B].getExcitedStateDipole(), r) *
                    _2PDM["PP"][state](A, B);
            eta.row(B) +=
                dipolePartial(monomers[A].getGroundStateDipole(), r) *
                    _2PDM["HP"][state](A, B) +
                dipolePartial(monomers[A].getTransitionDipole(), r) *
                    _2PDM["TP"][state](A, B) +
                dipolePartial(monomers[A].getExcitedStateDipole(), r) *
                    _2PDM["PP"][state](A, B);
          }

          // Eq. 146
          if (termStr == "R") {

            // HH
            auto HH = distancePartial(monomers[A].getGroundStateDipole(),
                                      monomers[B].getGroundStateDipole(), r);
            eta.row(A) -= HH * _2PDM["HH"][state](A, B);
            eta.row(B) += HH * _2PDM["HH"][state](A, B);

            // HT
            auto HT = distancePartial(monomers[A].getGroundStateDipole(),
                                      monomers[B].getTransitionDipole(), r);
            eta.row(A) -= HT * _2PDM["HT"][state](A, B);
            eta.row(B) += HT * _2PDM["HT"][state](A, B);

            // HP
            auto HP = distancePartial(monomers[A].getGroundStateDipole(),
                                      monomers[B].getExcitedStateDipole(), r);
            eta.row(A) -= HP * _2PDM["HP"][state](A, B);
            eta.row(B) += HP * _2PDM["HP"][state](A, B);

            // TH
            auto TH = distancePartial(monomers[A].getTransitionDipole(),
                                      monomers[B].getGroundStateDipole(), r);
            eta.row(A) -= TH * _2PDM["TH"][state](A, B);
            eta.row(B) += TH * _2PDM["TH"][state](A, B);

            // TT
            auto TT = distancePartial(monomers[A].getTransitionDipole(),
                                      monomers[B].getTransitionDipole(), r);
            eta.row(A) -= TT * _2PDM["TT"][state](A, B);
            eta.row(B) += TT * _2PDM["TT"][state](A, B);

            // TP
            auto TP = distancePartial(monomers[A].getTransitionDipole(),
                                      monomers[B].getExcitedStateDipole(), r);
            eta.row(A) -= TP * _2PDM["TP"][state](A, B);
            eta.row(B) += TP * _2PDM["TP"][state](A, B);

            // PH
            auto PH = distancePartial(monomers[A].getExcitedStateDipole(),
                                      monomers[B].getGroundStateDipole(), r);
            eta.row(A) -= PH * _2PDM["PH"][state](A, B);
            eta.row(B) += PH * _2PDM["PH"][state](A, B);

            // PT
            auto PT = distancePartial(monomers[A].getExcitedStateDipole(),
                                      monomers[B].getTransitionDipole(), r);
            eta.row(A) -= PT * _2PDM["PT"][state](A, B);
            eta.row(B) += PT * _2PDM["PT"][state](A, B);

            // PP
            auto PP = distancePartial(monomers[A].getExcitedStateDipole(),
                                      monomers[B].getExcitedStateDipole(), r);
            eta.row(A) -= PP * _2PDM["PP"][state](A, B);
            eta.row(B) += PP * _2PDM["PP"][state](A, B);
          }
        }
      }
      termEta.push_back(eta / 2.0);
    }
    dimerInteractionGradients.emplace(termStr, termEta);
  }

  return dimerInteractionGradients;
}

// here we contract the MC-VQE density matrices with the classical derivatives
// vector<stateGradient>
// stateGradients = vector<monomerGradient>
std::vector<std::vector<Eigen::MatrixXd>> MC_VQE::getNuclearGradients(
    std::map<std::string, std::vector<Eigen::MatrixXd>> &DM) {

  std::vector<std::vector<Eigen::MatrixXd>> gradients;

  // loop over states
  for (int state = 0; state < nStates; state++) {

    std::vector<Eigen::MatrixXd> stateGradients;
    // loop over chromophores
    for (int A = 0; A < nChromophores; A++) {

      auto nAtoms = monomers[A].getNumberOfAtoms();
      Eigen::MatrixXd monomerGradient = Eigen::MatrixXd::Zero(nAtoms, 3);

      monomerGradient +=
          DM["EH"][state](A) * monomers[A].getGroundStateEnergyGradient();
      monomerGradient +=
          DM["EP"][state](A) * monomers[A].getExcitedStateEnergyGradient();

      auto groundStateDipoleGradient =
          monomers[A].getGroundStateDipoleGradient();
      auto excitedStateDipoleGradient =
          monomers[A].getExcitedStateDipoleGradient();
      auto transitionDipoleGradient = monomers[A].getTransitionDipoleGradient();
      auto centerOfMassGradient = monomers[A].getCenterOfMassGradient();

      for (int i = 0; i < 3; i++) {
        monomerGradient += DM["MH"][state](A, i) * groundStateDipoleGradient[i];
        monomerGradient +=
            DM["MP"][state](A, i) * excitedStateDipoleGradient[i];
        monomerGradient += DM["MT"][state](A, i) * transitionDipoleGradient[i];
        monomerGradient += DM["R"][state](A, i) * centerOfMassGradient[i];
      }
      stateGradients.push_back(monomerGradient);
    }

    gradients.push_back(stateGradients);
  }

  return gradients;
}

} // namespace algorithm
} // namespace xacc
