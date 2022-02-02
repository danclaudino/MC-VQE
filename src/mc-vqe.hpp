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
 ******************************************************************************/
#ifndef XACC_ALGORITHM_MC_VQE_HPP_
#define XACC_ALGORITHM_MC_VQE_HPP_

#include "Algorithm.hpp"
#include "AlgorithmGradientStrategy.hpp"
#include "Monomer.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace xacc {

// Helper class
class Importable : public Identifiable {

public:
  Importable() = default;
  virtual void setEnergyDataPath(const std::string) = 0;
  virtual void setResponseDataPath(const std::string) { return; };
  virtual std::vector<Monomer> getMonomers(const int nChromophores) = 0;
};

namespace algorithm {

class MC_VQE : public Algorithm {

private:
  // gets time stamp
  double timer() const;

  // prints messages for given log level
  void logControl(const std::string message, const int level) const;

  // gets unrelaxed (Pauli) 1-body density matrix
  std::map<std::string, std::vector<Eigen::MatrixXd>>
  getUnrelaxed1PDM(const std::vector<double> &x) const;

  // gets unrelaxed (Pauli) 2-body density matrix
  std::map<std::string, std::vector<Eigen::MatrixXd>>
  getUnrelaxed2PDM(const std::vector<double> &x) const;

  // calls the VQE objective function
  double vqeWrapper(const std::shared_ptr<Observable> observable,
                    const std::shared_ptr<CompositeInstruction> kernel,
                    const std::vector<double> &x) const;

  // gets 1-body density matrix wrt VQE parameters
  std::map<std::string, std::vector<Eigen::MatrixXd>>
  getVQE1PDM(const std::vector<double> &,
             const std::vector<Eigen::MatrixXd> &) const;

  // gets 2-body density matrix wrt VQE parameters
  std::map<std::string, std::vector<Eigen::MatrixXd>>
  getVQE2PDM(const std::vector<double> &,
             const std::vector<Eigen::MatrixXd> &) const;

  // gets gradient of state energy wrt to state prep. angles
  Eigen::MatrixXd
  getStatePrepationAngleGradient(const Eigen::MatrixXd &,
                                 const std::vector<double> &) const;
  // implemented entanglers
  const std::vector<std::string> entanglers = {"default", "trotterized", "Ry"};

protected:
  Optimizer *optimizer;
  Accelerator *accelerator;
  HeterogeneousMap parameters;
  std::shared_ptr<AlgorithmGradientStrategy> gradientStrategy;

  // MC-VQE variables
  // default gradient strategy
  std::string gradientStrategyName = "parameter-shift";
  // default entangler type
  std::string entanglerType = "default";
  // vector of Monomers
  std::vector<Monomer> monomers;
  // number of chromophores
  int nChromophores;
  // true if the molecular system is cyclic
  bool isCyclic = false;
  // if false, will not compute interference matrix
  bool doInterference = true;
  // compute gradients
  bool doGradient = false;
  // CIS angles, states, and Hamiltonian
  Eigen::MatrixXd CISGateAngles, CISEigenstates, CISHamiltonian;
  // CIS eigenenergies
  Eigen::VectorXd CISEnergies;
  // Eigenvectors of subspace Hamiltonian
  mutable Eigen::MatrixXd subSpaceRotation;
  // MC energies (diagonal of subspace Hamiltonian)
  mutable Eigen::VectorXd diagonal;
  // AIEM Hamiltonian
  std::shared_ptr<Observable> observable;
  // # number of CIS states = nChromophores + 1
  int nStates;
  // path to file with quantum chemistry data
  std::string energyFilePath, responseFilePath;
  // entangler part of the circuit
  std::shared_ptr<CompositeInstruction> entangler;
  // start time of simulation
  std::chrono::system_clock::time_point start;
  // mc-vqe log file
  mutable std::ofstream logFile;
  // valid chromophore pairs (nearest neighbor)
  mutable std::map<int, std::vector<int>> pairs;
  // pi
  const double PI = xacc::constants::pi;

  // controls the level of printing
  int logLevel = 1;
  // controls whether TNQVM will print
  bool tnqvmLog = false;

  // MC-VQE methods

  // stores valid pairs of chromophores
  void setPairs() const;
  // compute CIS
  void computeCIS();
  // compute AIEM Hamiltonian
  void computeAIEMHamiltonian();
  // compute subspace, contracted Hamiltonian
  void computeSubspaceHamiltonian(Eigen::MatrixXd &,
                                  const std::vector<double> &) const;

  // constructs CIS state preparation circiuit
  std::shared_ptr<CompositeInstruction>
  statePreparationCircuit(const Eigen::VectorXd &stateAngles) const;

  // constructs entangler portion of MC-VQE circuit
  std::shared_ptr<CompositeInstruction> entanglerCircuit() const;

  // gets angles for states
  Eigen::MatrixXd
  statePreparationAngles(const Eigen::MatrixXd &coefficientMatrix) const;

  // response/gradient
  std::map<std::string, std::vector<Eigen::MatrixXd>>
  getUnrelaxedDensityMatrices(const std::vector<double> &x) const {
    auto monomerDM = getUnrelaxed1PDM(x);
    auto dimerDM = getUnrelaxed2PDM(x);
    monomerDM.insert(dimerDM.begin(), dimerDM.end());
    return monomerDM;
  };

  // gets VQE Lagrange multiplies
  std::vector<Eigen::MatrixXd>
  getVQEMultipliers(const std::vector<double> &) const;

  // gets VQE DMs
  std::map<std::string, std::vector<Eigen::MatrixXd>>
  getVQEDensityMatrices(const std::vector<double> &x,
                        std::vector<Eigen::MatrixXd> &multipliers) const {
    auto monomerDM = getVQE1PDM(x, multipliers);
    auto dimerDM = getVQE2PDM(x, multipliers);
    monomerDM.insert(dimerDM.begin(), dimerDM.end());
    return monomerDM;
  };

  // gets the Lagrange multiplies wrt to CIS-
  std::vector<Eigen::MatrixXd>
  getCRSMultipliers(const std::vector<double> &,
                    const std::vector<Eigen::MatrixXd> &) const;

  // gets DMs wrt to CIS
  std::map<std::string, std::vector<Eigen::MatrixXd>>
  getCRSDensityMatrices(std::vector<Eigen::MatrixXd> &) const;

  // gets the relaxed density matrices
  std::map<std::string, std::vector<Eigen::MatrixXd>> getRelaxedDensityMatrices(
      std::map<std::string, std::vector<Eigen::MatrixXd>> &,
      std::map<std::string, std::vector<Eigen::MatrixXd>> &,
      std::map<std::string, std::vector<Eigen::MatrixXd>> &) const;


  // gets DMs in the monomer basis
  std::map<std::string, std::vector<Eigen::MatrixXd>>
  getMonomerBasisDensityMatrices(
      std::map<std::string, std::vector<Eigen::MatrixXd>> &) const;

  // gets 1-body gradient contribution
  std::map<std::string, std::vector<Eigen::MatrixXd>> getMonomerGradient(
      std::map<std::string, std::vector<Eigen::MatrixXd>> &) const;

  // gets 2-body gradient contribution
  std::map<std::string, std::vector<Eigen::MatrixXd>>
  getDimerInteractionGradient(
      std::map<std::string, std::vector<Eigen::MatrixXd>> &) const;

  // gets gradient wrt nuclear displacement
  std::vector<std::vector<Eigen::MatrixXd>> getNuclearGradients(
      std::map<std::string, std::vector<Eigen::MatrixXd>> &) const;

public:
  bool initialize(const HeterogeneousMap &parameters) override;
  const std::vector<std::string> requiredParameters() const override;
  void execute(const std::shared_ptr<AcceleratorBuffer> buffer) const override;
  std::vector<double> execute(const std::shared_ptr<AcceleratorBuffer> buffer,
                              const std::vector<double> &parameters) override;
  const std::string name() const override { return "mc-vqe"; }
  const std::string description() const override { return ""; }
  DEFINE_ALGORITHM_CLONE(MC_VQE)
};
} // namespace algorithm
} // namespace xacc
#endif
