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
#include <vector>

namespace xacc {
namespace algorithm {

// This imports the data in the files
// top_dir/examples/2_qubit_datafile.txt
// top_dir/examples/18_qubit_datafile.txt
// top_dir/examples/60_qubit_datafile.txt
class ImportFromTestFile : public Importable {
private:
  std::vector<Monomer> monomers;

public:
  void import(const int nChromophores, const std::string dataDir) override {
    std::string dataPath;
    if (nChromophores == 2) {
      dataPath = dataDir + "2_qubit_datafile.txt";
    } else if (nChromophores <= 18) {
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
    Eigen::Vector3d tmpVec;
    int xyz, start;
    // scans output file and retrieve data
    for (int A = 0; A < nChromophores; A++) {

      // this is just the number label of the chromophore
      std::getline(file, line);
      std::getline(file, line);
      auto groundStateEnergy = std::stod(line.substr(line.find(":") + 1));
      std::getline(file, line);
      auto excitedStateEnergy = std::stod(line.substr(line.find(":") + 1));

      std::getline(file, line);
      tmp = line.substr(line.find(":") + 1);
      std::stringstream centerOfMassStream(tmp);
      Eigen::Vector3d centerOfMass = Eigen::Vector3d::Zero();
      xyz = 0;
      while (std::getline(centerOfMassStream, comp, ',')) {
        centerOfMass(xyz++) = std::stod(comp); // * ANGSTROM2BOHR;
      }

      std::getline(file, line);
      tmp = line.substr(line.find(":") + 1);
      Eigen::Vector3d groundStateDipole = Eigen::Vector3d::Zero();
      std::stringstream gsDipoleStream(tmp);
      xyz = 0;
      while (std::getline(gsDipoleStream, comp, ',')) {
        groundStateDipole(xyz++) = std::stod(comp); // * DEBYE2AU;
      }

      std::getline(file, line);
      tmp = line.substr(line.find(":") + 1);
      std::stringstream esDipoleStream(tmp);
      Eigen::Vector3d excitedStateDipole = Eigen::Vector3d::Zero();
      xyz = 0;
      while (std::getline(esDipoleStream, comp, ',')) {
        excitedStateDipole(xyz++) = std::stod(comp); // * DEBYE2AU;
      }

      std::getline(file, line);
      tmp = line.substr(line.find(":") + 1);
      std::stringstream tDipoleStream(tmp);
      xyz = 0;
      Eigen::Vector3d transitionDipole = Eigen::Vector3d::Zero();
      while (std::getline(tDipoleStream, comp, ',')) {
        transitionDipole(xyz++) = std::stod(comp);
      }

      Monomer m(groundStateEnergy, excitedStateEnergy, groundStateDipole,
                excitedStateDipole, transitionDipole, centerOfMass);
      monomers.push_back(m);
    }
    file.close();
    return;
  }

  std::vector<Monomer> getMonomers() override { return monomers; }
  const std::string name() const override { return "import-from-test-file"; }
  const std::string description() const override { return ""; }
};

} // namespace algorithm
} // namespace xacc
#endif
