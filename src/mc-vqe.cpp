/*******************************************************************************
 * Copyright (c) 2020 UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * and Eclipse Distribution License v1.0 which accompanies this
 * distribution. The Eclipse Public License is available at
 * http://www.eclipse.org/legal/epl-v10.html and the Eclipse Distribution
 *License is available at https://eclipse.org/org/documents/edl-v10.php
 *
 * Contributors:
 *   Daniel Claudino - initial API and implementation
 *******************************************************************************/
#include "mc-vqe.hpp"

#include "CompositeInstruction.hpp"
#include "Observable.hpp"
#include "PauliOperator.hpp"
#include "xacc_plugin.hpp"
#include "xacc_service.hpp"
#include <Eigen/Eigenvalues>
#include <iomanip>
#include <memory>
#include <string>

using namespace xacc;
using namespace xacc::quantum;

namespace xacc {
namespace algorithm {
bool MC_VQE::initialize(const HeterogeneousMap &parameters) {
  /** Checks for the required parameters and other optional keywords
   * @param[in] HeterogeneousMap A map of strings to keys
   */

  logFile.open("mcvqe.log", std::ofstream::out);
  start = std::chrono::high_resolution_clock::now();
  if (!parameters.pointerLikeExists<Accelerator>("accelerator")) {
    std::cout << "Acc was false\n";
    return false;
  }

  if (!parameters.pointerLikeExists<Optimizer>("optimizer")) {
    std::cout << "Opt was false\n";
    return false;
  }

  if (!parameters.keyExists<int>("nChromophores")) {
    std::cout << "Missing number of chromophores\n";
    return false;
  }

  if (!parameters.stringExists("data-path")) {
    std::cout << "Missing data file\n";
    return false;
  }

  optimizer = parameters.getPointerLike<Optimizer>("optimizer");
  accelerator = parameters.getPointerLike<Accelerator>("accelerator");
  nChromophores = parameters.get<int>("nChromophores");
  dataPath = parameters.getString("data-path");

  // This is to include the last entangler if the system is cyclic
  if (parameters.keyExists<bool>("cyclic")) {
    isCyclic = parameters.get<bool>("cyclic");
  } else {
    isCyclic = false;
  }

  // Determines the level of printing
  if (parameters.keyExists<int>("log-level")) {
    logLevel = parameters.get<int>("log-level");
  }

  // Turns TNQVM logging on/off
  if (parameters.keyExists<bool>("tnqvm-log")) {
    tnqvmLog = parameters.get<bool>("tnqvm-log");
  }

  // Controls whether interference matrix is computed
  // (For testing purposes)
  if (parameters.keyExists<bool>("interference")) {
    doInterference = parameters.get<bool>("interference");
  }

  nStates = nChromophores + 1;
  // here we get an instance of the Importable class and import the data
  // if this is not provided, it defaults to the test implementation
  std::shared_ptr<xacc::algorithm::Importable> import;
  if (parameters.stringExists("import")) {
    import = xacc::getService<xacc::algorithm::Importable>(
        parameters.getString("import"));
  } else {
    import =
        xacc::getService<xacc::algorithm::Importable>("import-from-test-file");
  }

  import->import(nChromophores, dataPath);
  monomers = import->getMonomers();

  // set pairs based on the connectivity
  setPairs();

  // compute Hamiltonians
  computeCIS();
  computeAIEM();

  // TODO : set up circuit gradients

  // Number of states to compute
  if (parameters.keyExists<int>("n-states")) {
    nStates = parameters.get<int>("n-states");
  }

  return true;
}

const std::vector<std::string> MC_VQE::requiredParameters() const {
  return {"optimizer", "accelerator", "nChromophores", "data-path"};
}

void MC_VQE::execute(const std::shared_ptr<AcceleratorBuffer> buffer) const {

  // entangledHamiltonian stores the Hamiltonian matrix elements
  // in the basis of MC states
  Eigen::MatrixXd entangledHamiltonian =
      Eigen::MatrixXd::Zero(nStates, nStates);

  // all CIS states share the same parameterized entangler gates
  auto entangler = entanglerCircuit();
  // number of parameters to be optimized
  auto nOptParams = entangler->nVariables();
  // Only store the diagonal when it lowers the average energy
  diagonal = Eigen::VectorXd::Zero(nStates);

  // store circuit depth, # of gates and pass to buffer
  int depth = 0, nGates = 0;

  // optimization history
  std::vector<double> optHistory;

  double oldAverageEnergy = 0.0;
  // f is the objective function
  OptFunction f(
      [&, this](const std::vector<double> &x, std::vector<double> &dx) {
        // MC-VQE minimizes the average energy over all MC states
        double averageEnergy = 0.0;

        // we take the gradient of the average energy as the average over
        // the gradients of each state
        std::vector<double> averageGrad(x.size());

        // loop over states
        for (int state = 0; state < nStates; state++) {

          // retrieve CIS state preparation instructions and add entangler
          auto kernel = statePreparationCircuit(CISGateAngles.col(state));
          kernel->addVariables(entangler->getVariables());
          for (auto &inst : entangler->getInstructions()) {
            kernel->addInstruction(inst);
          }
          logControl("Circuit preparation for state # " +
                         std::to_string(state) + " [" +
                         std::to_string(timer()) + " s]",
                     2);

          if (depth == 0) {
            depth = kernel->depth();
            nGates = kernel->nInstructions();
          }

          // Call VQE objective function
          logControl(
              "Calling vqe::execute() [" + std::to_string(timer()) + " s]", 2);
          if (tnqvmLog) {
            xacc::set_verbose(true);
          }
          auto energy = vqeWrapper(observable, kernel, x);
          if (tnqvmLog) {
            xacc::set_verbose(false);
          }
          logControl("Completed vqe::execute() [" + std::to_string(timer()) +
                         " s]",
                     2);
          logControl("State # " + std::to_string(state) +
                         " energy = " + std::to_string(energy) + " [" +
                         std::to_string(timer()) + " s]",
                     2);

          // state energy goes to the diagonal of entangledHamiltonian
          diagonal(state) = energy;
          averageEnergy += energy;
        }

        logControl("Energy of all states computed for the current entangler [" +
                       std::to_string(timer()) + " s]",
                   1);

        // Average energy at the end of current iteration
        averageEnergy /= nStates;

        // only store the MC energies if they lower the average energy
        if (averageEnergy < oldAverageEnergy) {
          oldAverageEnergy = averageEnergy;
          logControl(
              "Updated average energy [" + std::to_string(timer()) + " s]", 1);
          entangledHamiltonian.diagonal() = diagonal;
          logControl("Updated Hamiltonian diagonal [" +
                         std::to_string(timer()) + " s]",
                     1);
        }

        logControl("Optimization iteration finished [" +
                       std::to_string(timer()) + " s]",
                   2);

        // store current info
        std::stringstream ss;
        ss << "E(" << (!x.empty() ? std::to_string(x[0]) : "");
        for (int i = 1; i < x.size(); i++)
          ss << "," << x[i];
        ss << ") = " << std::setprecision(12) << averageEnergy;
        logControl(ss.str() + "\n", 2);

        optHistory.push_back(averageEnergy);

        logControl("Iteration " + std::to_string(optHistory.size()) + " = " +
                       std::to_string(averageEnergy),
                   1);
        // logControl(std::to_string(diagonal(1) - diagonal(0)), 1);

        return averageEnergy;
      },
      nOptParams);

  // run optimization
  logControl("Starting the MC-VQE optimization [" + std::to_string(timer()) +
                 " s]\n",
             1);
  auto result = optimizer->optimize(f);
  logControl("MC-VQE optimization complete [" + std::to_string(timer()) + " s]",
             1);

  buffer->addExtraInfo("opt-average-energy", ExtraInfo(result.first));
  buffer->addExtraInfo("circuit-depth", ExtraInfo(depth));
  buffer->addExtraInfo("n-gates", ExtraInfo(nGates));
  buffer->addExtraInfo("opt-params", ExtraInfo(result.second));
  logControl("Persisted info in the buffer [" + std::to_string(timer()) + " s]",
             1);

  if (doInterference) {
    // now construct interference states and observe Hamiltonian
    auto optimizedEntangler = result.second;
    logControl(
        "Computing Hamiltonian matrix elements in the interference state basis",
        1);
    for (int stateA = 0; stateA < nStates - 1; stateA++) {
      for (int stateB = stateA + 1; stateB < nStates; stateB++) {

        // |+> = (|A> + |B>)/sqrt(2)
        auto plusInterferenceCircuit = statePreparationCircuit(
            (CISGateAngles.col(stateA) + CISGateAngles.col(stateB)) /
            std::sqrt(2));
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
        auto plusTerm =
            vqeWrapper(observable, plusInterferenceCircuit, optimizedEntangler);
        if (tnqvmLog) {
          xacc::set_verbose(false);
        }

        // |-> = (|A> - |B>)/sqrt(2)
        auto minusInterferenceCircuit = statePreparationCircuit(
            (CISGateAngles.col(stateA) - CISGateAngles.col(stateB)) /
            std::sqrt(2));
        minusInterferenceCircuit->addVariables(entangler->getVariables());
        for (auto &inst : entangler->getInstructions()) {
          minusInterferenceCircuit->addInstruction(inst);
        }

        // vqe call to compute the expectation value of the AEIM Hamiltonian
        // in the plusInterferenceCircuit with optimal entangler parameters
        // It does NOT perform any optimization
        if (tnqvmLog) {
          xacc::set_verbose(true);
        }
        auto minusTerm = vqeWrapper(observable, minusInterferenceCircuit,
                                    optimizedEntangler);
        if (tnqvmLog) {
          xacc::set_verbose(false);
        }

        // add the new matrix elements to entangledHamiltonian
        entangledHamiltonian(stateA, stateB) =
            (plusTerm - minusTerm) / std::sqrt(2);
        entangledHamiltonian(stateB, stateA) =
            entangledHamiltonian(stateA, stateB);
      }
    }

    logControl("Computed interference basis Hamiltonian matrix elements [" +
                   std::to_string(timer()) + " s]",
               1);

    logControl("Diagonalizing entangled Hamiltonian", 1);

    // Diagonalizing the entangledHamiltonian gives the energy spectrum
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> EigenSolver(
        entangledHamiltonian);
    auto MC_VQE_Energies = EigenSolver.eigenvalues();
    auto MC_VQE_States = EigenSolver.eigenvectors();

    logControl("Diagonalized entangled Hamiltonian [" +
                   std::to_string(timer()) + " s]",
               1);

    std::stringstream ss;
    ss << "MC-VQE energy spectrum";
    for (auto e : MC_VQE_Energies) {
      ss << "\n" << std::setprecision(9) << e;
    }

    buffer->addExtraInfo("opt-spectrum", ExtraInfo(ss.str()));
    logControl(ss.str(), 1);

    logControl("MC-VQE simulation finished [" + std::to_string(timer()) + " s]",
               1);
  }

  logFile.close();
  return;
}

std::vector<double>
MC_VQE::execute(const std::shared_ptr<AcceleratorBuffer> buffer,
                const std::vector<double> &x) {

  Eigen::MatrixXd entangledHamiltonian =
      Eigen::MatrixXd::Zero(nStates, nStates);

  // all CIS states share the same entangler gates
  auto entangler = entanglerCircuit();

  // store circuit depth and pass to buffer
  int depth, nGates;

  // MC-VQE minimizes the average energy over all MC states
  double averageEnergy = 0.0;
  for (int state = 0; state < nStates; state++) { // loop over states

    // prepare CIS state
    auto kernel = statePreparationCircuit(CISGateAngles.col(state));
    // add entangler variables
    kernel->addVariables(entangler->getVariables());
    for (auto &inst : entangler->getInstructions()) {
      kernel->addInstruction(inst); // append entangler gates
    }

    depth = kernel->depth();
    nGates = kernel->nInstructions();

    logControl("Printing circuit for state #" + std::to_string(state), 3);
    logControl(kernel->toString(), 3);

    logControl("Printing instructions for state #" + std::to_string(state), 4);
    // loop over CompositeInstructions from observe
    if (tnqvmLog) {
      xacc::set_verbose(true);
    }
    auto energy = vqeWrapper(observable, kernel, x);
    if (tnqvmLog) {
      xacc::set_verbose(false);
    }

    // state energy goes to the diagonal of entangledHamiltonian
    entangledHamiltonian(state, state) = energy;
    averageEnergy += energy / nStates;
    logControl("State # " + std::to_string(state) + " energy " +
                   std::to_string(energy),
               2);
  }

  buffer->addExtraInfo("opt-average-energy", averageEnergy);
  buffer->addExtraInfo("circuit-depth", ExtraInfo(depth));
  buffer->addExtraInfo("n-gates", ExtraInfo(nGates));
  if (doInterference) {
    // now construct interference states and observe Hamiltonian
    logControl(
        "Computing Hamiltonian matrix elements in the interference state basis",
        1);
    for (int stateA = 0; stateA < nStates - 1; stateA++) {
      for (int stateB = stateA + 1; stateB < nStates; stateB++) {

        // |+> = (|A> + |B>)/sqrt(2)
        auto plusInterferenceCircuit = statePreparationCircuit(
            (CISGateAngles.col(stateA) + CISGateAngles.col(stateB)) /
            std::sqrt(2));
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
        auto minusInterferenceCircuit = statePreparationCircuit(
            (CISGateAngles.col(stateA) - CISGateAngles.col(stateB)) /
            std::sqrt(2));
        minusInterferenceCircuit->addVariables(entangler->getVariables());
        for (auto &inst : entangler->getInstructions()) {
          minusInterferenceCircuit->addInstruction(inst);
        }

        // vqe call to compute the expectation value of the AEIM Hamiltonian
        // in the plusInterferenceCircuit with optimal entangler parameters
        // It does NOT perform any optimization
        if (tnqvmLog) {
          xacc::set_verbose(true);
        }
        auto minusTerm = vqeWrapper(observable, minusInterferenceCircuit, x);
        if (tnqvmLog) {
          xacc::set_verbose(false);
        }

        // add the new matrix elements to entangledHamiltonian
        entangledHamiltonian(stateA, stateB) =
            (plusTerm - minusTerm) / std::sqrt(2);
        entangledHamiltonian(stateB, stateA) =
            entangledHamiltonian(stateA, stateB);
      }
    }

    logControl("Diagonalizing entangled Hamiltonian", 1);
    // Diagonalizing the entangledHamiltonian gives the energy spectrum
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> EigenSolver(
        entangledHamiltonian);
    auto MC_VQE_Energies = EigenSolver.eigenvalues();
    auto MC_VQE_States = EigenSolver.eigenvectors();

    std::stringstream ss;
    ss << "MC-VQE energy spectrum";
    for (auto e : MC_VQE_Energies) {
      ss << "\n" << std::setprecision(9) << e;
    }

    buffer->addExtraInfo("opt-spectrum", ExtraInfo(ss.str()));
    logControl(ss.str(), 1);

    std::vector<double> spectrum(MC_VQE_Energies.data(),
                                 MC_VQE_Energies.data() +
                                     MC_VQE_Energies.size());

    return spectrum;
  } else {
    return {};
  }
}

void MC_VQE::setPairs() {
  // stores the indices of the valid chromophore pairs
  // assuming the chromophores can only be in a
  // cyclic or linear arrangement

  if (isCyclic) {
    for (int A = 0; A < nChromophores; A++) {
      if (A == 0) {
        pairs[A] = {A + 1, nChromophores - 1};
      } else if (A == nChromophores - 1) {
        pairs[A] = {A - 1, 0};
      } else {
        pairs[A] = {A - 1, A + 1};
      }
    }
  } else {
    for (int A = 0; A < nChromophores; A++) {
      if (A == 0) {
        pairs[A] = {A + 1};
      } else if (A == nChromophores - 1) {
        pairs[A] = {A - 1};
      } else {
        pairs[A] = {A - 1, A + 1};
      }
    }
  }
  return;
}

void MC_VQE::computeCIS() {

  // CIS matrix elements in the nChromophore two-state basis
  CISHamiltonian = Eigen::MatrixXd::Zero(nStates, nStates);

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

void MC_VQE::computeAIEM() {

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

Eigen::MatrixXd
MC_VQE::statePreparationAngles(const Eigen::MatrixXd CoefficientMatrix) {

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

void MC_VQE::logControl(const std::string message, const int level) const {
  /** Function that controls the level of printing
   * @param[in] message This is what is to be printed
   * @param[in] level message is printed if level is above logLevel
   */

  if (logLevel >= level) {
    xacc::set_verbose(true);
    xacc::info(message);
    logFile << message << "\n";
    xacc::set_verbose(false);
  }
  return;
}

// This is just a wrapper to compute <H^n> with VQE::execute(q, {})
double MC_VQE::vqeWrapper(const std::shared_ptr<Observable> observable,
                          const std::shared_ptr<CompositeInstruction> kernel,
                          const std::vector<double> x) const {

  auto q = xacc::qalloc(nChromophores);
  auto vqe = xacc::getAlgorithm("vqe", {{"observable", observable},
                                        {"accelerator", accelerator},
                                        {"ansatz", kernel}});
  return vqe->execute(q, x)[0];
}

double MC_VQE::timer() const {

  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::duration<double>>(now - start)
      .count();
}

} // namespace algorithm
} // namespace xacc

// REGISTER_ALGORITHM(xacc::algorithm::MC_VQE)
