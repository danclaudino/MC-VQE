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
  readData();
  computeAIEMAndCIS();

  // Number of states to compute (< nChromophores + 1)
  // (For testing purposes)
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
          if (tnqvmLog) {
            xacc::set_verbose(true);
          }
          logControl(
              "Calling vqe::execute() [" + std::to_string(timer()) + " s]", 2);
          auto energy = vqeWrapper(observable, kernel, x);
          logControl("Completed vqe::execute() [" + std::to_string(timer()) +
                         " s]",
                     2);
          if (tnqvmLog) {
            xacc::set_verbose(false);
          }
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

std::shared_ptr<CompositeInstruction>
MC_VQE::statePreparationCircuit(const Eigen::VectorXd &angles) const {
  /** Constructs the circuit that prepares a CIS state
   * @param[in] angles Angles to parameterize CIS state
   */

  // Provider to create IR for CompositeInstruction, aka circuit
  auto provider = xacc::getIRProvider("quantum");
  // allocate memory for circuit
  auto statePreparationInstructions = provider->createComposite("mcvqeCircuit");

  // Beginning of CIS state preparation
  //
  // Ry "pump"
  auto ry = provider->createInstruction("Ry", {0}, {angles(0)});
  statePreparationInstructions->addInstruction(ry);

  // Fy gates = Ry(-theta/2)-CZ-Ry(theta/2) = Ry(-theta/2)-H-CNOT-H-Ry(theta/2)
  //
  // |A>------------------------o------------------------
  //                            |
  // |B>--[Ry(-angles(i)/2)]-H-[X]-H-[Ry(+angles(i)/2)]--
  //
  for (int i = 1; i < nChromophores; i++) {

    std::size_t control = i - 1, target = i;
    // Ry(-theta/2)
    auto ry_minus =
        provider->createInstruction("Ry", {target}, {-angles(2 * i - 1) / 2});
    statePreparationInstructions->addInstruction(ry_minus);

    auto hadamard1 = provider->createInstruction("H", {target});
    statePreparationInstructions->addInstruction(hadamard1);

    auto cnot = provider->createInstruction("CNOT", {control, target});
    statePreparationInstructions->addInstruction(cnot);

    auto hadamard2 = provider->createInstruction("H", {target});
    statePreparationInstructions->addInstruction(hadamard2);

    // Ry(+theta/2)
    auto ry_plus =
        provider->createInstruction("Ry", {target}, {angles(2 * i) / 2});
    statePreparationInstructions->addInstruction(ry_plus);
  }

  // Wall of CNOTs
  for (int i = nChromophores - 2; i >= 0; i--) {
    for (int j = nChromophores - 1; j > i; j--) {

      std::size_t control = j, target = i;
      auto cnot = provider->createInstruction("CNOT", {control, target});
      statePreparationInstructions->addInstruction(cnot);
    }
  }
  // end of CIS state preparation
  return statePreparationInstructions;
}

std::shared_ptr<CompositeInstruction> MC_VQE::entanglerCircuit() const {
  /** Constructs the entangler part of the circuit
   */

  // Create CompositeInstruction for entangler
  auto provider = xacc::getIRProvider("quantum");
  auto entanglerInstructions = provider->createComposite("mcvqeCircuit");

  // structure of entangler
  // does not implement the first two Ry to remove redundancies in the circuit
  // and thus reducing the number of variational parameters
  //
  // |A>--[Ry(x0)]--o--[Ry(x2)]--o--[Ry(x4)]-
  //                |            |
  // |B>--[Ry(x1)]--x--[Ry(x3)]--x--[Ry(x5)]-
  //
  auto entanglerGate = [&](const std::size_t control, const std::size_t target,
                           int paramCounter) {
    /** Lambda function to construct the entangler gates
     * Interleaves CNOT(control, target) and Ry rotations
     *
     * @param[in] control Index of the control/source qubit
     * @param[in] target Index of the target qubit
     * @param[in] paramCounter Variable that keeps track of and indexes the
     * circuit parameters
     */

    auto cnot1 = provider->createInstruction("CNOT", {control, target});
    entanglerInstructions->addInstruction(cnot1);

    auto ry1 = provider->createInstruction(
        "Ry", {control},
        {InstructionParameter("x" + std::to_string(paramCounter))});
    entanglerInstructions->addVariable("x" + std::to_string(paramCounter));
    entanglerInstructions->addInstruction(ry1);

    auto ry2 = provider->createInstruction(
        "Ry", {target},
        {InstructionParameter("x" + std::to_string(paramCounter + 1))});
    entanglerInstructions->addVariable("x" + std::to_string(paramCounter + 1));
    entanglerInstructions->addInstruction(ry2);

    auto cnot2 = provider->createInstruction("CNOT", {control, target});
    entanglerInstructions->addInstruction(cnot2);

    auto ry3 = provider->createInstruction(
        "Ry", {control},
        {InstructionParameter("x" + std::to_string(paramCounter + 2))});
    entanglerInstructions->addVariable("x" + std::to_string(paramCounter + 2));
    entanglerInstructions->addInstruction(ry3);

    auto ry4 = provider->createInstruction(
        "Ry", {target},
        {InstructionParameter("x" + std::to_string(paramCounter + 3))});
    entanglerInstructions->addVariable("x" + std::to_string(paramCounter + 3));
    entanglerInstructions->addInstruction(ry4);
  };

  // placing the first Ry's in the circuit
  int paramCounter = 0;
  for (int i = 0; i < nChromophores; i++) {
    // std::size_t q = i;
    auto ry = provider->createInstruction(
        "Ry", {(std::size_t)i},
        {InstructionParameter("x" + std::to_string(paramCounter))});
    entanglerInstructions->addVariable("x" + std::to_string(paramCounter));
    entanglerInstructions->addInstruction(ry);

    paramCounter++;
  }

  // placing the entanglerGates in the circuit
  for (int layer : {0, 1}) {
    for (int i = layer; i < nChromophores - layer; i += 2) {
      std::size_t control = i, target = i + 1;
      entanglerGate(control, target, paramCounter);
      paramCounter += NPARAMSENTANGLER;
    }
  }

  // if the molecular system is cyclic, we need to add this last entangler
  if (isCyclic) {
    std::size_t control = nChromophores - 1, target = 0;
    entanglerGate(control, target, paramCounter);
  }

  // end of entangler circuit construction
  return entanglerInstructions;
}

void MC_VQE::readData() {
  /** Function to process the quantum chemistry data into CIS state preparation
   * angles and the AIEM Hamiltonian.
   *
   * Excited state refers to the first excited state.
   * Ref1 = PRL 122, 230401 (2019)
   * Ref2 = Supplemental Material for Ref2
   * Ref3 = arXiv:1906.08728v1
   */

  // instantiate necessary ingredients for quantum chemistry input
  // energiesGS = ground state energies
  // energiesES = excited state energies
  // dipoleGS = ground state dipole moment vectors
  // dipoleES = excited state dipole moment vectors
  // dipoleT = transition (between ground and excited states) dipole moment
  // com = center of mass of each chromophore
  energiesGS = Eigen::VectorXd::Zero(nChromophores);
  energiesES = Eigen::VectorXd::Zero(nChromophores);
  dipoleGS = Eigen::MatrixXd::Zero(nChromophores, 3);
  dipoleES = Eigen::MatrixXd::Zero(nChromophores, 3);
  dipoleT = Eigen::MatrixXd::Zero(nChromophores, 3);
  centerOfMass = Eigen::MatrixXd::Zero(nChromophores, 3);

  std::ifstream file(dataPath);
  if (file.bad()) {
    xacc::error("Cannot access data file.");
    return;
  }

  std::string line, tmp, comp;
  int xyz, start;
  // scans output file and retrieve data
  for (int A = 0; A < nChromophores; A++) {

    // this is just the number label of the chromophore
    std::getline(file, line);
    std::getline(file, line);
    energiesGS(A) = std::stod(line.substr(line.find(":") + 1));
    std::getline(file, line);
    energiesES(A) = std::stod(line.substr(line.find(":") + 1));

    std::getline(file, line);
    tmp = line.substr(line.find(":") + 1);
    std::stringstream comStream(tmp);
    xyz = 0;
    while (std::getline(comStream, comp, ',')) {
      centerOfMass(A, xyz) = std::stod(comp);
      xyz++;
    }

    std::getline(file, line);
    tmp = line.substr(line.find(":") + 1);
    std::stringstream gsDipoleStream(tmp);
    xyz = 0;
    while (std::getline(gsDipoleStream, comp, ',')) {
      dipoleGS(A, xyz) = std::stod(comp);
      xyz++;
    }

    std::getline(file, line);
    tmp = line.substr(line.find(":") + 1);
    std::stringstream esDipoleStream(tmp);
    xyz = 0;
    while (std::getline(esDipoleStream, comp, ',')) {
      dipoleES(A, xyz) = std::stod(comp);
      xyz++;
    }

    std::getline(file, line);
    tmp = line.substr(line.find(":") + 1);
    std::stringstream tDipoleStream(tmp);
    xyz = 0;
    while (std::getline(tDipoleStream, comp, ',')) {
      dipoleT(A, xyz) = std::stod(comp);
      xyz++;
    }
  }
  file.close();

  centerOfMass *= ANGSTROM2BOHR; // angstrom to bohr
  dipoleGS *= DEBYE2AU; // D to a.u.
  dipoleES *= DEBYE2AU;

  return;
}

void MC_VQE::computeAIEMAndCIS() {

  auto twoBodyH = [&](const Eigen::VectorXd mu_A, const Eigen::VectorXd mu_B,
                      const Eigen::VectorXd r_AB) {
    /** Lambda function to compute the two-body AIEM Hamiltonian matrix elements
     * (Ref.2 Eq. 67)
     * @param[in] mu_A Some dipole moment vector (gs/es/t) from chromophore A
     * @param[in] mu_B Some dipole moment vector (gs/es/t) from chromophore B
     * @param[in] r_AB Vector between A and B
     */
    auto d_AB = r_AB.norm(); // Distance between A and B
    auto n_AB = r_AB / d_AB; // Normal vector along r_AB
    return (mu_A.dot(mu_B) - 3.0 * mu_A.dot(n_AB) * mu_B.dot(n_AB)) /
           std::pow(d_AB, 3.0); // return matrix element
  };

  // stores the indices of the valid chromophore pairs
  for (int A = 0; A < nChromophores; A++) {
    if (A == 0 && isCyclic) {
      pairs[A] = {A + 1, nChromophores - 1};
    } else if (A == nChromophores - 1) {
      pairs[A] = {A - 1, 0};
    } else {
      pairs[A] = {A - 1, A + 1};
    }
  }

  // Ref2 Eq. 20 and 21
  //auto S_A = (energiesGS + energiesES) / 2.0;
  //auto D_A = (energiesGS - energiesES) / 2.0;
  // sum of dipole moments, coordinate-wise
  auto dipoleSum = (dipoleGS + dipoleES) / 2.0;
  // difference of dipole moments, coordinate-wise
  auto dipoleDiff = (dipoleGS - dipoleES) / 2.0;

  Eigen::VectorXd Z_A = (energiesGS - energiesES) / 2.0;
  Eigen::VectorXd X_A = Eigen::VectorXd::Zero(nChromophores);
  Eigen::MatrixXd XX_AB = Eigen::MatrixXd::Zero(nChromophores, nChromophores);
  Eigen::MatrixXd XZ_AB = Eigen::MatrixXd::Zero(nChromophores, nChromophores);
  Eigen::MatrixXd ZX_AB = Eigen::MatrixXd::Zero(nChromophores, nChromophores);
  Eigen::MatrixXd ZZ_AB = Eigen::MatrixXd::Zero(nChromophores, nChromophores);

  // Compute the AIEM Hamiltonian
  PauliOperator hamiltonian;
  double E = 0.0; // = S_A.sum(), but this only shifts the eigenvalues
  for (int A = 0; A < nChromophores; A++) {

    for (int B : pairs[A]) {

      E += 0.5 * twoBodyH(dipoleSum.row(A), dipoleSum.row(B),
                          centerOfMass.row(A) - centerOfMass.row(B));
      // E += 0.5 * twoBodyH(dipoleSum.row(B), dipoleSum.row(A), centerOfMass.row(B) -
      // centerOfMass.row(A));

      X_A(A) += 0.5 * twoBodyH(dipoleT.row(A), dipoleSum.row(B),
                               centerOfMass.row(A) - centerOfMass.row(B));
      X_A(A) += 0.5 * twoBodyH(dipoleSum.row(B), dipoleT.row(A),
                               centerOfMass.row(B) - centerOfMass.row(A));

      Z_A(A) += 0.5 * twoBodyH(dipoleSum.row(A), dipoleDiff.row(B),
                               centerOfMass.row(A) - centerOfMass.row(B));
      Z_A(A) += 0.5 * twoBodyH(dipoleDiff.row(B), dipoleSum.row(A),
                               centerOfMass.row(B) - centerOfMass.row(A));

      XX_AB(A, B) =
          twoBodyH(dipoleT.row(A), dipoleT.row(B), centerOfMass.row(A) - centerOfMass.row(B));
      XZ_AB(A, B) =
          twoBodyH(dipoleT.row(A), dipoleDiff.row(B), centerOfMass.row(A) - centerOfMass.row(B));
      ZX_AB(A, B) =
          twoBodyH(dipoleDiff.row(A), dipoleT.row(B), centerOfMass.row(A) - centerOfMass.row(B));
      ZZ_AB(A, B) = twoBodyH(dipoleDiff.row(A), dipoleDiff.row(B),
                             centerOfMass.row(A) - centerOfMass.row(B));

      hamiltonian += PauliOperator({{A, "X"}, {B, "X"}}, XX_AB(A, B));
      hamiltonian += PauliOperator({{A, "X"}, {B, "Z"}}, XZ_AB(A, B));
      hamiltonian += PauliOperator({{A, "Z"}, {B, "X"}}, ZX_AB(A, B));
      hamiltonian += PauliOperator({{A, "Z"}, {B, "Z"}}, ZZ_AB(A, B));
    }

    hamiltonian += PauliOperator({{A, "Z"}}, Z_A(A));
    hamiltonian += PauliOperator({{A, "X"}}, X_A(A));
  }
  hamiltonian += PauliOperator(E);

  // Done with the AIEM Hamiltonian just need a pointer for it
  observable = std::make_shared<PauliOperator>(hamiltonian);

  logControl("Computed AIEM Hamiltonian [" + std::to_string(timer()) + " s]",
             1);

  // CIS matrix elements in the nChromophore two-state basis
  CISHamiltonian = Eigen::MatrixXd::Zero(nStates, nStates);

  // E_ref
  auto E_ref = E + Z_A.sum() + 0.5 * ZZ_AB.sum();
  CISHamiltonian(0, 0) = E_ref;

  // diagonal singles-singles
  for (int A = 0; A < nChromophores; A++) {
    CISHamiltonian(A + 1, A + 1) = E_ref - 2.0 * Z_A(A);
    for (int B : pairs[A]) {
      CISHamiltonian(A + 1, A + 1) -= ZZ_AB(A, B) + ZZ_AB(B, A);
    }
  }

  // reference-singles off diagonal
  for (int A = 0; A < nChromophores; A++) {
    CISHamiltonian(A + 1, 0) = X_A(A);
    for (int B : pairs[A]) {
      CISHamiltonian(A + 1, 0) += 0.5 * (XZ_AB(A, B) + ZX_AB(B, A));
    }
    CISHamiltonian(0, A + 1) = CISHamiltonian(A + 1, 0);
  }

  // singles-singles off-diagonal
  for (int A = 0; A < nChromophores; A++) {
    for (int B : pairs[A]) {
      CISHamiltonian(A + 1, B + 1) = XX_AB(A, B);
    }
  }

  // Diagonalizing the CISHamiltonian
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> EigenSolver(CISHamiltonian);
  CISEnergies = EigenSolver.eigenvalues();
  CISEigenstates = EigenSolver.eigenvectors();

  logControl("Computed CIS parameters [" + std::to_string(timer()) + " s]",
             1);

  CISGateAngles.resize(nChromophores, nStates);
  CISGateAngles = statePreparationAngles(CISEigenstates);

  // end of preProcessing
  return;
}

Eigen::MatrixXd
MC_VQE::statePreparationAngles(const Eigen::MatrixXd CoefficientMatrix) {

  Eigen::MatrixXd gateAngles =
      Eigen::MatrixXd::Zero(2 * nChromophores - 1, nStates);
  // Computing the CIS state preparation angles (Ref3 Eqs. 60-61)

  for (int state = 0; state < nStates; state++) {
    for (int angle = 0; angle < nChromophores; angle++) {

      double partialCoeffNorm = CoefficientMatrix.col(state)
                                    .segment(angle, nChromophores - angle + 1)
                                    .norm();

      if (angle == 0) {
        gateAngles(angle, state) =
            std::acos(CoefficientMatrix(angle, state) / partialCoeffNorm);

      } else {
        gateAngles(2 * angle - 1, state) =
            std::acos(CoefficientMatrix(angle, state) / partialCoeffNorm);
        gateAngles(2 * angle, state) =
            std::acos(CoefficientMatrix(angle, state) / partialCoeffNorm);
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
    xacc::info(message);
    logFile << message << "\n";
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

REGISTER_ALGORITHM(xacc::algorithm::MC_VQE)
