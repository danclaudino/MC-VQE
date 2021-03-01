#include "Circuit.hpp"
#include "PauliOperator.hpp"
#include "mc-vqe.hpp"
#include <Eigen/Eigenvalues>
#include <complex>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace xacc;
using namespace xacc::quantum;

namespace xacc {
namespace algorithm {

void MC_VQE::computeCIS() {

  // CIS matrix elements in the nChromophore two-state basis
  CISHamiltonian = Eigen::MatrixXd::Zero(nChromophores + 1, nChromophores + 1);

  // diagonal singles-singles
  for (int A = 0; A < nChromophores; A++) {
    CISHamiltonian(A + 1, A + 1) = -2.0 * monomers[A].getD();
    for (int B : pairs[A]) {
      CISHamiltonian(A + 1, A + 1) -= 2.0 * (monomers[A].getZ(monomers[B]) +
                                             monomers[A].getZZ(monomers[B]) +
                                             monomers[B].getZZ(monomers[A]));
    }
  }

  // reference-singles off diagonal
  for (int A = 0; A < nChromophores; A++) {
    for (int B : pairs[A]) {
      CISHamiltonian(A + 1, 0) += monomers[A].getX(monomers[B]) +
                                  monomers[A].getXZ(monomers[B]) +
                                  monomers[B].getZX(monomers[A]);
    }
  }

  // singles-singles off-diagonal
  for (int A = 0; A < nChromophores; A++) {
    for (int B : pairs[A]) {
      CISHamiltonian(A + 1, B + 1) = 2.0 * monomers[A].getXX(monomers[B]);
    }
  }

  // Diagonalizing the CISHamiltonian
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> EigenSolver(CISHamiltonian);
  CISEnergies = EigenSolver.eigenvalues();
  CISEigenstates = EigenSolver.eigenvectors();

  logControl("Computed CIS parameters [" + std::to_string(timer()) + " s]", 1);

  CISGateAngles.resize(nChromophores, nStates);
  CISGateAngles = statePreparationAngles(CISEigenstates);

  // end of preProcessing
  return;
}

void MC_VQE::computeAIEMHamiltonian() {

  // Compute the AIEM Hamiltonian
  PauliOperator hamiltonian;
  for (int A = 0; A < nChromophores; A++) {

    double X = 0.0, Z = monomers[A].getD();

    for (int B : pairs[A]) {

      X += monomers[A].getX(monomers[B]);
      Z += monomers[A].getZ(monomers[B]);

      hamiltonian +=
          PauliOperator({{A, "X"}, {B, "X"}}, monomers[A].getXX(monomers[B]));
      hamiltonian +=
          PauliOperator({{A, "X"}, {B, "Z"}}, monomers[A].getXZ(monomers[B]));
      hamiltonian +=
          PauliOperator({{A, "Z"}, {B, "X"}}, monomers[A].getZX(monomers[B]));
      hamiltonian +=
          PauliOperator({{A, "Z"}, {B, "Z"}}, monomers[A].getZZ(monomers[B]));
    }

    hamiltonian += PauliOperator({{A, "Z"}}, Z);
    hamiltonian += PauliOperator({{A, "X"}}, X);
  }
  // Done with the AIEM Hamiltonian just need a pointer for it
  observable = std::make_shared<PauliOperator>(hamiltonian);

  logControl("Computed AIEM Hamiltonian [" + std::to_string(timer()) + " s]",
             1);

  return;
}

void MC_VQE::computeSubspaceHamiltonian(Eigen::MatrixXd &entangledHamiltonian,
                                        const std::vector<double> &x) const {

  for (int stateA = 0; stateA < nStates - 1; stateA++) {
    for (int stateB = stateA + 1; stateB < nStates; stateB++) {

      // |+> = (|A> + |B>)/sqrt(2)
      Eigen::VectorXd plusCoefficients =
          (CISEigenstates.col(stateA) + CISEigenstates.col(stateB)) / sqrt(2.0);
      auto plusGateAngles = statePreparationAngles(plusCoefficients);
      auto plusInterferenceCircuit = statePreparationCircuit(plusGateAngles);
      plusInterferenceCircuit->addVariables(entangler->getVariables());
      for (auto &inst : entangler->getInstructions()) {
        plusInterferenceCircuit->addInstruction(inst);
      }

      // vqe call to compute the expectation value of the AEIM Hamiltonian
      // in the plusInterferenceCircuit with optimal entangler parameters
      // It does NOT perform any optimization
      if (tnqvmLog) {
        xacc::set_verbose(true);
      }
      auto plusTerm = vqeWrapper(observable, plusInterferenceCircuit, x);
      if (tnqvmLog) {
        xacc::set_verbose(false);
      }

      // |-> = (|A> - |B>)/sqrt(2)
      Eigen::VectorXd minusCoefficients =
          (CISEigenstates.col(stateA) - CISEigenstates.col(stateB)) / sqrt(2.0);
      auto minusGateAngles = statePreparationAngles(minusCoefficients);
      auto minusInterferenceCircuit = statePreparationCircuit(minusGateAngles);
      minusInterferenceCircuit->addVariables(entangler->getVariables());
      for (auto &inst : entangler->getInstructions()) {
        minusInterferenceCircuit->addInstruction(inst);
      }

      if (tnqvmLog) {
        xacc::set_verbose(true);
      }
      auto minusTerm = vqeWrapper(observable, minusInterferenceCircuit, x);
      if (tnqvmLog) {
        xacc::set_verbose(false);
      }

      // add the new matrix elements to entangledHamiltonian
      entangledHamiltonian(stateA, stateB) = (plusTerm - minusTerm) / 2.0;
      entangledHamiltonian(stateB, stateA) =
          entangledHamiltonian(stateA, stateB);
    }
  }

  return;
}

Eigen::MatrixXd
MC_VQE::statePreparationAngles(const Eigen::MatrixXd &CoefficientMatrix) {

  Eigen::MatrixXd gateAngles =
      Eigen::MatrixXd::Zero(2 * nChromophores - 1, nStates);

  // Computing the CIS state preparation angles (Ref3 Eqs. 60-61)
  // We need to multiply the angles by 2 to match Rob's results
  // Probably because his simulator implements rotations as
  // e^(-i * theta * Y), while we do e^(-i * theta/2 * Y)
  for (int state = 0; state < nStates; state++) {
    for (int angle = 0; angle < nChromophores; angle++) {

      double partialCoeffNorm = CoefficientMatrix.col(state)
                                    .segment(angle, nChromophores - angle + 1)
                                    .norm();

      if (angle == 0) {
        gateAngles(angle, state) =
            2.0 * std::acos(CoefficientMatrix(angle, state) / partialCoeffNorm);

      } else {
        gateAngles(2 * angle - 1, state) =
            2.0 * std::acos(CoefficientMatrix(angle, state) / partialCoeffNorm);
        gateAngles(2 * angle, state) =
            2.0 * std::acos(CoefficientMatrix(angle, state) / partialCoeffNorm);
      }
    }

    // multiply the last gate angle by the sign of the
    // last coefficient for the state
    if (CoefficientMatrix(Eigen::last, state) < 0.0) {
      gateAngles(2 * nChromophores - 2, state) *= -1.0;
      gateAngles(2 * nChromophores - 3, state) *= -1.0;
    }
  }

  return gateAngles;
}

Eigen::VectorXd
MC_VQE::statePreparationAngles(const Eigen::VectorXd &CoefficientVector) const {

  Eigen::VectorXd gateAngles = Eigen::VectorXd::Zero(2 * nChromophores - 1);

  // Computing the CIS state preparation angles (Ref3 Eqs. 60-61)
  // We need to multiply the angles by 2 to match Rob's results
  // Probably because his simulator implements rotations as
  // e^(-i * theta * Y), while we do e^(-i * theta/2 * Y)
  for (int angle = 0; angle < nChromophores; angle++) {

    double partialCoeffNorm =
        CoefficientVector.segment(angle, nChromophores - angle + 1).norm();

    if (angle == 0) {
      gateAngles(angle) =
          2.0 * std::acos(CoefficientVector(angle) / partialCoeffNorm);

    } else {
      gateAngles(2 * angle - 1) =
          2.0 * std::acos(CoefficientVector(angle) / partialCoeffNorm);
      gateAngles(2 * angle) =
          2.0 * std::acos(CoefficientVector(angle) / partialCoeffNorm);
    }
  }

  // multiply the last gate angle by the sign of the
  // last coefficient for the state
  if (CoefficientVector(Eigen::last) < 0.0) {
    gateAngles(2 * nChromophores - 2) *= -1.0;
    gateAngles(2 * nChromophores - 3) *= -1.0;
  }

  return gateAngles;
}

}
}