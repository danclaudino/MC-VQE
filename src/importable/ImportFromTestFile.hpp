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

#include "Monomer.hpp"
#include <iomanip>
#include <map>
#include <string>
#include <vector>

namespace xacc {

class ImportFromTestFile : public Importable {

private:
  std::ifstream energyFile, responseFile;

public:
  void setEnergyDataPath(const std::string energyFilePath) override {

    energyFile.open(energyFilePath);
    if (energyFile.fail()) {
      xacc::error("Cannot access data file.");
      return;
    }
    return;
  }

  void setResponseDataPath(const std::string responseFilePath) override {

    responseFile.open(responseFilePath);
    if (responseFile.fail()) {
      xacc::error("Cannot access data file.");
      return;
    }

    return;
  }

  std::vector<Monomer> getMonomers(const int nChromophores) override {

    auto getVector = [&](const std::string line) {
      std::string tmp = line.substr(line.find(":") + 1), comp;
      std::stringstream ss(tmp);
      Eigen::Vector3d vec;
      int xyz = 0;
      while (std::getline(ss, comp, ',')) {
        vec(xyz++) = std::stod(comp);
      }
      return vec;
    };

    auto getGradient = [&](const int nAtoms) {
      std::string line;
      Eigen::MatrixXd gradient(nAtoms, 3);
      for (int i = 0; i < nAtoms; i++) {
        std::getline(responseFile, line);
        gradient.row(i) = getVector(line);
      }
      return gradient;
    };

    std::string line;
    std::vector<Monomer> monomers;
    // scans output file and retrieve data
    for (int A = 0; A < nChromophores; A++) {

      Monomer m;

      // this is just the number label of the chromophore
      std::getline(energyFile, line);
      std::getline(energyFile, line);
      auto groundStateEnergy = std::stod(line.substr(line.find(":") + 1));
      m.setGroundStateEnergy(groundStateEnergy);

      std::getline(energyFile, line);
      auto excitedStateEnergy = std::stod(line.substr(line.find(":") + 1));
      m.setExcitedStateEnergy(excitedStateEnergy);

      std::getline(energyFile, line);
      m.setCenterOfMass(getVector(line));

      std::getline(energyFile, line);
      m.setGroundStateDipole(getVector(line));

      std::getline(energyFile, line);
      m.setExcitedStateDipole(getVector(line));

      std::getline(energyFile, line);
      m.setTransitionDipole(getVector(line));

      // read in data for response part
      if (responseFile.is_open()) {
        std::getline(responseFile, line);
        std::getline(responseFile, line);
        auto nAtoms = std::stoi(line.substr(line.find(":") + 1));
        m.setNumberOfAtoms(nAtoms);

        std::getline(responseFile, line);
        m.setGroundStateEnergyGradient(getGradient(nAtoms));

        std::getline(responseFile, line);
        m.setExcitedStateEnergyGradient(getGradient(nAtoms));

        std::vector<Eigen::MatrixXd> centerOfMassGradient;
        for (int i = 0; i < 3; i++) {
          std::getline(responseFile, line);
          centerOfMassGradient.push_back(getGradient(nAtoms));
        }
        m.setCenterOfMassGradient(centerOfMassGradient);

        std::vector<Eigen::MatrixXd> groundStateDipoleGradient;
        for (int i = 0; i < 3; i++) {
          std::getline(responseFile, line);
          groundStateDipoleGradient.push_back(getGradient(nAtoms));
        }
        m.setGroundStateDipoleGradient(groundStateDipoleGradient);

        std::vector<Eigen::MatrixXd> excitedStateDipoleGradient;
        for (int i = 0; i < 3; i++) {
          std::getline(responseFile, line);
          excitedStateDipoleGradient.push_back(getGradient(nAtoms));
        }
        m.setExcitedStateDipoleGradient(excitedStateDipoleGradient);

        std::vector<Eigen::MatrixXd> transitionDipoleGradient;
        for (int i = 0; i < 3; i++) {
          std::getline(responseFile, line);
          transitionDipoleGradient.push_back(getGradient(nAtoms));
        }
        m.setTransitionDipoleGradient(transitionDipoleGradient);

      }

      monomers.push_back(m);
    }

    energyFile.close();
    if (responseFile.is_open()) {
      responseFile.close();
    }

    return monomers;
  }

  const std::string name() const override { return "import-from-test-file"; }
  const std::string description() const override { return ""; }
};

} // namespace xacc
#endif
