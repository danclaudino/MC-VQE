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
#ifndef XACC_ALGORITHM_MC_VQE_HPP_
#define XACC_ALGORITHM_MC_VQE_HPP_

#include "Algorithm.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <fstream>

namespace xacc {
namespace algorithm {
class MC_VQE : public Algorithm {
protected:
  Optimizer *optimizer;
  Accelerator *accelerator;
  HeterogeneousMap parameters;

  // MC-VQE variables

  // number of chromophores
  int nChromophores;
  // true if the molecular system is cyclic
  bool isCyclic;
  // if false, will not compute interference matrix
  bool doInterference = true;
  // state preparation angles
  Eigen::MatrixXd CISGateAngles, CISEigenstates, CISHamiltonian;
  Eigen::VectorXd CISEnergies;
  mutable Eigen::MatrixXd subSpaceRotation;
  mutable Eigen::VectorXd diagonal;
  // AIEM Hamiltonian
  std::shared_ptr<Observable> observable;
  // # number of CIS states = nChromophores + 1
  int nStates;
  // # of parameters in a single entangler
  // after merging adjacent Ry gates
  const int NPARAMSENTANGLER = 4;
  // angstrom to bohr
  const double ANGSTROM2BOHR = 1.8897161646320724;
  // D to a.u.
  const double DEBYE2AU = 0.393430307;
  // pi/4
  const double PI_4 = xacc::constants::pi / 4.0;
  // path to file with quantum chemistry data
  std::string dataPath;
  // entangler part of the circuit
  std::shared_ptr<CompositeInstruction> entangler;
  // start time of simulation
  std::chrono::system_clock::time_point start;
  // mc-vqe log file
  mutable std::ofstream logFile;
  // valid chromophore pairs (nearest neighbor)
  std::map<int, std::vector<int>> pairs;

  // constructs CIS state preparation circiuit
  std::shared_ptr<CompositeInstruction>
  statePreparationCircuit(const Eigen::VectorXd &stateAngles) const;

  // constructs entangler portion of MC-VQE circuit
  std::shared_ptr<CompositeInstruction> entanglerCircuit() const;

  // reads quantum chemistry data
  void readData();
  Eigen::VectorXd energiesGS;
  Eigen::VectorXd energiesES;
  Eigen::MatrixXd dipoleGS;
  Eigen::MatrixXd dipoleES;
  Eigen::MatrixXd dipoleT;
  Eigen::MatrixXd centerOfMass;

  // compute AIEM Hamiltonian and CIS
  void computeAIEMAndCIS();

  // gets angles from state coefficients
  Eigen::MatrixXd
  statePreparationAngles(const Eigen::MatrixXd CoefficientMatrix);

  // calls the VQE objective function
  double vqeWrapper(const std::shared_ptr<Observable> observable,
                    const std::shared_ptr<CompositeInstruction> kernel,
                    const std::vector<double> x) const;

  // gets time stamp
  double timer() const;

  // controls the level of printing
  int logLevel = 1;
  bool tnqvmLog = false;
  void logControl(const std::string message, const int level) const;

  // response/gradient
  std::vector<Eigen::VectorXd> vqeMultipliers;
  std::vector<Eigen::MatrixXd> cpCRSMultipliers;
  std::vector<Eigen::VectorXd> getUnrelaxed1PDM(const std::string termStr,
                                                const std::vector<double> x);

  std::vector<Eigen::MatrixXd> getUnrelaxed2PDM(const std::string termStr,
                                                const std::vector<double> x);

  void getVQEMultipliers(const std::vector<double> x);

  Eigen::VectorXd getVQE1PDM(const std::string termStr,
                             const std::vector<double> x);

  Eigen::MatrixXd getVQE2PDM(const std::string termStr,
                             const std::vector<double> x);

  void getCRSMultipliers(const std::vector<double> x);

  std::vector<Eigen::VectorXd> getCRS1PDM(const std::string termStr);
  std::vector<Eigen::MatrixXd> getCRS2PDM(const std::string termStr);
  std::vector<Eigen::VectorXd> getMonomerGradient(const std::string, const std::vector<Eigen::VectorXd>);
std::vector<Eigen::MatrixXd>
getDimerDipoleGradient(const std::string,
                               const std::vector<Eigen::MatrixXd>);

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
