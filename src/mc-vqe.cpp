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

#include "PauliOperator.hpp"
#include <Eigen/Eigenvalues>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

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
    xacc::error("Acc was false");
    return false;
  }

  if (!parameters.pointerLikeExists<Optimizer>("optimizer")) {
    xacc::error("Opt was false");
    return false;
  }

  if (!parameters.keyExists<int>("nChromophores")) {
    xacc::error("Missing number of chromophores");
    return false;
  }

  if (!parameters.stringExists("energy-data-path")) {
    xacc::error("Missing data file");
    return false;
  }

  optimizer = parameters.getPointerLike<Optimizer>("optimizer");
  accelerator = parameters.getPointerLike<Accelerator>("accelerator");
  nChromophores = parameters.get<int>("nChromophores");
  energyFilePath = parameters.getString("energy-data-path");

  // This is to include the last entangler if the system is cyclic
  if (parameters.keyExists<bool>("cyclic")) {
    isCyclic = parameters.get<bool>("cyclic");
  }

  // get entangler
  if (parameters.stringExists("entangler")) {
    auto tmpEntangler = parameters.getString("entangler");
    if (!xacc::container::contains(entanglers, tmpEntangler)) {
      xacc::warning(entanglerType + " not implemented. Using default.");
    } else {
      entanglerType = tmpEntangler;
    }
  } else {
    xacc::warning("Using default SO(4) entangler");
  }
  entangler = entanglerCircuit();

  // Determines the level of printing
  if (parameters.keyExists<int>("log-level")) {
    logLevel = parameters.get<int>("log-level");
  }

  // Turns TNQVM logging on/off
  if (parameters.keyExists<bool>("tnqvm-log")) {
    tnqvmLog = parameters.get<bool>("tnqvm-log");
  }

  // Controls whether the interference matrix is computed
  // mostly for testing purposes
  if (parameters.keyExists<bool>("interference")) {
    doInterference = parameters.get<bool>("interference");
  }

  // boolean to control whether to compute gradients
  if (parameters.keyExists<bool>("mc-vqe-gradient")) {
    doGradient = parameters.get<bool>("mc-vqe-gradient");
  }

  // Number of states to compute
  if (parameters.keyExists<int>("n-states")) {
    nStates = parameters.get<int>("n-states");
  } else {
    nStates = nChromophores + 1;
  }

  // here we get an instance of the Importable class and import the data
  // if this is not provided, it defaults to the test implementation
  std::shared_ptr<xacc::Importable> import;
  if (parameters.stringExists("import")) {
    import = xacc::getService<xacc::Importable>(parameters.getString("import"));
  } else {
    import = xacc::getService<xacc::Importable>("import-from-test-file");
  }
  import->setEnergyDataPath(energyFilePath);

  // this is for the gradients
  if (parameters.stringExists("response-data-path")) {
    responseFilePath = parameters.getString("response-data-path");
    import->setResponseDataPath(responseFilePath);
  }

  // get monomer data
  monomers = import->getMonomers(nChromophores);

  // Convert Debye to a.u.
  if (parameters.keyExists<bool>("debye-to-au")) {
    for (auto &m : monomers) {
      m.isDipoleInDebye(parameters.get<bool>("debye-to-au"));
    }
  }

  // Convert Angstrom to a.u.
  if (parameters.keyExists<bool>("angstrom-to-au")) {
    for (auto &m : monomers) {
      m.isCenterOfMassInAngstrom(parameters.get<bool>("angstrom-to-au"));
    }
  }

  // set pairs based on the connectivity
  setPairs();

  // compute Hamiltonians
  computeCIS();
  computeAIEMHamiltonian();

  // we may need to come back here and understand the gradients better
  if (optimizer->isGradientBased()) {
    if (parameters.stringExists("gradient-strategy")) {
      gradientStrategyName = parameters.getString("gradient-strategy");
    } else {
      xacc::warning(
          "Gradient-based optimization defaults to parameter-shift strategy");
    }
    gradientStrategy =
        xacc::getService<AlgorithmGradientStrategy>(gradientStrategyName);
    gradientStrategy->initialize({{"observable", observable}});
  }

  return true;
}

const std::vector<std::string> MC_VQE::requiredParameters() const {
  return {"accelerator", "nChromophores", "energy-data-path"};
}

void MC_VQE::execute(const std::shared_ptr<AcceleratorBuffer> buffer) const {

  // entangledHamiltonian stores the Hamiltonian matrix elements
  // in the basis of MC states/interference basis
  Eigen::MatrixXd entangledHamiltonian =
      Eigen::MatrixXd::Zero(nStates, nStates);

  // number of parameters to be optimized

  // Only store the diagonal when it lowers the average energy
  diagonal = Eigen::VectorXd::Zero(nStates);

  // store circuit depth, # of gates and pass to buffer
  int depth = 0, nGates = 0;

  // optimization history
  std::vector<double> optHistory;

  // all CIS states share the same parameterized entangler gates
  auto nOptParams = entangler->nVariables();

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

          std::vector<double> stateGradient(x.size());

          // retrieve CIS state preparation instructions and add entangler
          auto kernel = statePreparationCircuit(CISGateAngles.col(state));
          kernel->addVariables(entangler->getVariables());
          kernel->addInstructions(entangler->getInstructions());

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

          // gather circuits for gradients
          std::vector<std::shared_ptr<CompositeInstruction>> gradFsToExec;
          if (gradientStrategy) {

            gradFsToExec = gradientStrategy->getGradientExecutions(kernel, x);

            xacc::info("Number of gradient circuit implemetation: " +
                       std::to_string(gradFsToExec.size()));

            // if gradient is numerical, we need F(0)
            if (gradientStrategy->isNumerical()) {
              auto identityCoeff =
                  std::real(std::dynamic_pointer_cast<PauliOperator>(observable)
                                ->getTerms()["I"]
                                .coeff());
              gradientStrategy->setFunctionValue(energy - identityCoeff);
            }

            // run gradient circuits
            auto tmpBuffer = xacc::qalloc(buffer->size());
            accelerator->execute(tmpBuffer, gradFsToExec);
            auto buffers = tmpBuffer->getChildren();

            // compute gradient vector
            gradientStrategy->compute(stateGradient, buffers);

            for (int i = 0; i < x.size(); i++) {
              dx[i] += stateGradient[i] / nStates;
            }
          }

          // state energy goes to the diagonal of entangledHamiltonian
          diagonal(state) = energy;
          averageEnergy += energy;
        }

        if (gradientStrategy) {

          double sum = 0.0;
          for (auto x : dx)
            sum += x * x;

          std::stringstream ss;
          ss << "G(" << (!dx.empty() ? std::to_string(dx[0]) : "");
          for (int i = 1; i < dx.size(); i++)
            ss << "," << dx[i];
          ss << ") = " << sqrt(sum);
          logControl(ss.str() + "\n", 2);
        }

        logControl("Energy of all states computed for the current entangler [" +
                       std::to_string(timer()) + " s]",
                   1);

        // Average energy at the end of current iteration
        averageEnergy /= nStates;

        // only store the MC energies if they lower the average energy
        if (averageEnergy < oldAverageEnergy) {
          oldAverageEnergy = averageEnergy;
          entangledHamiltonian.diagonal() = diagonal;
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
        logControl(std::to_string(diagonal(1) - diagonal(0)), 1);

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

  if (doInterference) {
    // now construct interference states and observe Hamiltonian
    auto optimizedEntangler = result.second;
    logControl("Computing Hamiltonian in the interference state basis", 1);
    computeSubspaceHamiltonian(entangledHamiltonian, result.second);
    logControl("Computed interference basis Hamiltonian matrix elements [" +
                   std::to_string(timer()) + " s]",
               1);

    logControl("Diagonalizing entangled Hamiltonian", 1);

    // Diagonalizing the entangledHamiltonian gives the energy spectrum
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> EigenSolver(
        entangledHamiltonian);
    auto MC_VQE_Energies = EigenSolver.eigenvalues();
    subSpaceRotation = EigenSolver.eigenvectors();

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

  if (doInterference) {
    // now construct interference states and observe Hamiltonian
    logControl(
        "Computing Hamiltonian matrix elements in the interference state basis",
        1);

    computeSubspaceHamiltonian(entangledHamiltonian, x);
    logControl("Diagonalizing entangled Hamiltonian", 1);
    // Diagonalizing the entangledHamiltonian gives the energy spectrum
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> EigenSolver(
        entangledHamiltonian);
    auto MC_VQE_Energies = EigenSolver.eigenvalues();
    subSpaceRotation = EigenSolver.eigenvectors();

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

    if (doGradient) {

      /*
      std::map<std::string, std::vector<Eigen::MatrixXd>> dm;

      Eigen::MatrixXd X1 = Eigen::VectorXd::Zero(nStates);
      Eigen::MatrixXd Z1 = Eigen::VectorXd::Zero(nStates);
      Eigen::MatrixXd XX1 = Eigen::MatrixXd::Zero(nStates, nStates);
      Eigen::MatrixXd XZ1 = Eigen::MatrixXd::Zero(nStates, nStates);
      Eigen::MatrixXd ZX1 = Eigen::MatrixXd::Zero(nStates, nStates);
      Eigen::MatrixXd ZZ1 = Eigen::MatrixXd::Zero(nStates, nStates);

      X1 << -0.07114968, -0.46273434;
      Z1 << 0.97345163, 0.94304295;
      XX1 << 0, 0.04008718, 0.04008718, 0;
      XZ1 << 0, -0.12594704, -0.35449684, 0;
      ZX1 = XZ1.transpose();
      ZZ1 << 0, 0.89717335, 0.89717335, 0;

      Eigen::MatrixXd X2 = Eigen::VectorXd::Zero(nStates);
      Eigen::MatrixXd Z2 = Eigen::VectorXd::Zero(nStates);
      Eigen::MatrixXd XX2 = Eigen::MatrixXd::Zero(nStates, nStates);
      Eigen::MatrixXd XZ2 = Eigen::MatrixXd::Zero(nStates, nStates);
      Eigen::MatrixXd ZX2 = Eigen::MatrixXd::Zero(nStates, nStates);
      Eigen::MatrixXd ZZ2 = Eigen::MatrixXd::Zero(nStates, nStates);

      X2 << -0.21860722, 0.26865738;
      Z2 << -0.42482695, 0.41315969;
      XX2 << 0, -0.89910805, -0.89910805, 0;
      XZ2 << 0, -0.00911019, -0.1114055, 0;
      ZX2 = XZ2.transpose();
      ZZ2 << 0, -0.96768438, -0.96768438, 0;

      dm["X"] = {X1, X2};
      dm["Z"] = {Z1, Z2};
      dm["XX"] = {XX1, XX2};
      dm["XZ"] = {XZ1, XZ2};
      dm["ZX"] = {ZX1, ZX2};
      dm["ZZ"] = {ZZ1, ZZ2};
      */

      auto unrelaxedDensityMatrices = getUnrelaxedDensityMatrices(x);
      auto vqeMultipliers = getVQEMultipliers(x);
      auto vqeDensityMatrices = getVQEDensityMatrices(x, vqeMultipliers);
      auto crsMultipliers = getCRSMultipliers(x, vqeMultipliers);
      auto crsDensityMatrices = getCRSDensityMatrices(crsMultipliers);
      auto relaxedDensityMatrices = getRelaxedDensityMatrices(
          unrelaxedDensityMatrices, vqeDensityMatrices, crsDensityMatrices);

      auto monomerDensityMatrices =
          getMonomerBasisDensityMatrices(relaxedDensityMatrices);

      auto monomerGradient = getMonomerGradient(relaxedDensityMatrices);

      auto dimerGradient = getDimerInteractionGradient(monomerDensityMatrices);

      monomerGradient.insert(dimerGradient.begin(), dimerGradient.end());

      auto nuclearGradient = getNuclearGradients(monomerGradient);
    }
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
    kernel->addInstructions(entangler->getInstructions());

    if (state == 0) {
      depth = kernel->depth();
      nGates = kernel->nInstructions();
    }

    logControl("Printing circuit for state #" + std::to_string(state), 3);
    logControl(kernel->toString(), 3);

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

    computeSubspaceHamiltonian(entangledHamiltonian, x);
    logControl("Diagonalizing entangled Hamiltonian", 1);
    // Diagonalizing the entangledHamiltonian gives the energy spectrum
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> EigenSolver(
        entangledHamiltonian);
    auto MC_VQE_Energies = EigenSolver.eigenvalues();
    subSpaceRotation = EigenSolver.eigenvectors();

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

    if (doGradient) {

      /*
      std::map<std::string, std::vector<Eigen::MatrixXd>> dm;

      Eigen::MatrixXd X1 = Eigen::VectorXd::Zero(nStates);
      Eigen::MatrixXd Z1 = Eigen::VectorXd::Zero(nStates);
      Eigen::MatrixXd XX1 = Eigen::MatrixXd::Zero(nStates, nStates);
      Eigen::MatrixXd XZ1 = Eigen::MatrixXd::Zero(nStates, nStates);
      Eigen::MatrixXd ZX1 = Eigen::MatrixXd::Zero(nStates, nStates);
      Eigen::MatrixXd ZZ1 = Eigen::MatrixXd::Zero(nStates, nStates);

      X1 << -0.07114968, -0.46273434;
      Z1 << 0.97345163, 0.94304295;
      XX1 << 0, 0.04008718, 0.04008718, 0;
      XZ1 << 0, -0.12594704, -0.35449684, 0;
      ZX1 = XZ1.transpose();
      ZZ1 << 0, 0.89717335, 0.89717335, 0;

      Eigen::MatrixXd X2 = Eigen::VectorXd::Zero(nStates);
      Eigen::MatrixXd Z2 = Eigen::VectorXd::Zero(nStates);
      Eigen::MatrixXd XX2 = Eigen::MatrixXd::Zero(nStates, nStates);
      Eigen::MatrixXd XZ2 = Eigen::MatrixXd::Zero(nStates, nStates);
      Eigen::MatrixXd ZX2 = Eigen::MatrixXd::Zero(nStates, nStates);
      Eigen::MatrixXd ZZ2 = Eigen::MatrixXd::Zero(nStates, nStates);

      X2 << -0.21860722, 0.26865738;
      Z2 << -0.42482695, 0.41315969;
      XX2 << 0, -0.89910805, -0.89910805, 0;
      XZ2 << 0, -0.00911019, -0.1114055, 0;
      ZX2 = XZ2.transpose();
      ZZ2 << 0, -0.96768438, -0.96768438, 0;

      dm["X"] = {X1, X2};
      dm["Z"] = {Z1, Z2};
      dm["XX"] = {XX1, XX2};
      dm["XZ"] = {XZ1, XZ2};
      dm["ZX"] = {ZX1, ZX2};
      dm["ZZ"] = {ZZ1, ZZ2};
      */

      auto unrelaxedDensityMatrices = getUnrelaxedDensityMatrices(x);
      auto vqeMultipliers = getVQEMultipliers(x);
      auto vqeDensityMatrices = getVQEDensityMatrices(x, vqeMultipliers);
      auto crsMultipliers = getCRSMultipliers(x, vqeMultipliers);
      auto crsDensityMatrices = getCRSDensityMatrices(crsMultipliers);
      auto relaxedDensityMatrices = getRelaxedDensityMatrices(
          unrelaxedDensityMatrices, vqeDensityMatrices, crsDensityMatrices);

      auto monomerDensityMatrices =
          getMonomerBasisDensityMatrices(relaxedDensityMatrices);

      auto monomerGradient = getMonomerGradient(relaxedDensityMatrices);

      auto dimerGradient = getDimerInteractionGradient(monomerDensityMatrices);

      monomerGradient.insert(dimerGradient.begin(), dimerGradient.end());

      auto nuclearGradient = getNuclearGradients(monomerGradient);
    }
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
                          const std::vector<double> &x) const {

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
