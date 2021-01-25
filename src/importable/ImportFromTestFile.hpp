/*******************************************************************************
 * Copyright (c) 2021 UT-Battelle, LLC.
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
#ifndef XACC_IMPORT_FROM_TEST_FILE_HPP_
#define XACC_IMPORT_FROM_TEST_FILE_HPP_

#include "mc-vqe.hpp"
#include "xacc.hpp"
#include "xacc_plugin.hpp"

namespace xacc {
namespace algorithm {

// This imports the data in the files
// top_dir/examples/18_qubit_datafile.txt and
// top_dir/examples/60_qubit_datafile.txt
class ImportFromTestFile : public Importable {
private:
  Eigen::VectorXd groundStateEnergies;
  Eigen::VectorXd excitedStateEnergies;
  Eigen::MatrixXd groundStateDipoles;
  Eigen::MatrixXd excitedStateDipoles;
  Eigen::MatrixXd transitionDipoles;
  // center of mass
  Eigen::MatrixXd centerOfMass;

  // angstrom to bohr
  const double ANGSTROM2BOHR = 1.8897161646320724;
  // D to a.u.
  const double DEBYE2AU = 0.393430307;

public:
  void import(const int nChromophores, const std::string dataDir) override {

    // instantiate necessary ingredients for quantum chemistry input
    // com = center of mass of each chromophore
    groundStateEnergies = Eigen::VectorXd::Zero(nChromophores);
    excitedStateEnergies = Eigen::VectorXd::Zero(nChromophores);
    groundStateDipoles = Eigen::MatrixXd::Zero(nChromophores, 3);
    excitedStateDipoles = Eigen::MatrixXd::Zero(nChromophores, 3);
    transitionDipoles = Eigen::MatrixXd::Zero(nChromophores, 3);
    centerOfMass = Eigen::MatrixXd::Zero(nChromophores, 3);

    std::string dataPath;
    if (nChromophores <= 18) {
      dataPath = dataDir + "18_qubit_datafile.txt";
    } else {
      dataPath = dataDir + "60_qubit_datafile.txt";
    }

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
      groundStateEnergies(A) = std::stod(line.substr(line.find(":") + 1));
      std::getline(file, line);
      excitedStateEnergies(A) = std::stod(line.substr(line.find(":") + 1));

      std::getline(file, line);
      tmp = line.substr(line.find(":") + 1);
      std::stringstream centerOfMasstream(tmp);
      xyz = 0;
      while (std::getline(centerOfMasstream, comp, ',')) {
        centerOfMass(A, xyz) = std::stod(comp);
        xyz++;
      }

      std::getline(file, line);
      tmp = line.substr(line.find(":") + 1);
      std::stringstream gsDipoleStream(tmp);
      xyz = 0;
      while (std::getline(gsDipoleStream, comp, ',')) {
        groundStateDipoles(A, xyz) = std::stod(comp);
        xyz++;
      }

      std::getline(file, line);
      tmp = line.substr(line.find(":") + 1);
      std::stringstream esDipoleStream(tmp);
      xyz = 0;
      while (std::getline(esDipoleStream, comp, ',')) {
        excitedStateDipoles(A, xyz) = std::stod(comp);
        xyz++;
      }

      std::getline(file, line);
      tmp = line.substr(line.find(":") + 1);
      std::stringstream tDipoleStream(tmp);
      xyz = 0;
      while (std::getline(tDipoleStream, comp, ',')) {
        transitionDipoles(A, xyz) = std::stod(comp);
        xyz++;
      }
    }
    file.close();

    centerOfMass *= ANGSTROM2BOHR;
    groundStateDipoles *= DEBYE2AU;
    excitedStateDipoles *= DEBYE2AU;

    return;
  }

  Eigen::VectorXd getGroundStateEnergies() override {
    return groundStateEnergies;
  }

  Eigen::VectorXd getExcitedStateEnergies() override {
    return excitedStateEnergies;
  }

  Eigen::MatrixXd getGroundStateDipoles() override {
    return groundStateDipoles;
  }

  Eigen::MatrixXd getExcitedStateDipoles() override {
    return excitedStateDipoles;
  }

  Eigen::MatrixXd getTransitionDipoles() override { return transitionDipoles; }

  Eigen::MatrixXd getCenterOfMass() override { return centerOfMass; }

  const std::string name() const override { return "import-from-test-file"; }
  const std::string description() const override { return ""; }
};

} // namespace algorithm
} // namespace xacc
#endif

// REGISTER_PLUGIN(xacc::ImportReadFromFile, xacc::Importable)