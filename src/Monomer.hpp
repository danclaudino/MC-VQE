#pragma once

#include <Eigen/Dense>
#include <map>

class Monomer {
private:
  int _nAtoms;
  const double ANGSTROM2BOHR = 1.8897161646320724, DEBYE2AU = 0.393430307;
  double _groundStateEnergy, _excitedStateEnergy;
  Eigen::Vector3d _groundStateDipole, _excitedStateDipole, _transitionDipole,
      _centerOfMass;

  Eigen::MatrixXd _groundStateEnergyGradient, _excitedStateEnergyGradient;
  std::vector<Eigen::MatrixXd> _groundStateDipoleGradient,
      _excitedStateDipoleGradient, _transitionDipoleGradient,
      _centerOfMassGradient;

  double twoBodyH(const Eigen::VectorXd &mu_A, const Eigen::VectorXd &mu_B,
                  const Eigen::VectorXd &r_AB);

public:
  Monomer() = default;

  void isDipoleInDebye(const bool);
  void isCenterOfMassInAngstrom(const bool);
  double getS() const;
  double getD() const;
  double getE() const;

  // energy
  void setGroundStateEnergy(const double);
  void setExcitedStateEnergy(const double);
  void setGroundStateDipole(const Eigen::Vector3d);
  void setExcitedStateDipole(const Eigen::Vector3d);
  void setTransitionDipole(const Eigen::Vector3d);
  void setCenterOfMass(const Eigen::Vector3d);

  // gradient
  void setNumberOfAtoms(const int);
  void setGroundStateEnergyGradient(const Eigen::MatrixXd &);
  void setExcitedStateEnergyGradient(const Eigen::MatrixXd &);
  void setCenterOfMassGradient(const std::vector<Eigen::MatrixXd> &);
  void setGroundStateDipoleGradient(const std::vector<Eigen::MatrixXd> &);
  void setExcitedStateDipoleGradient(const std::vector<Eigen::MatrixXd> &);
  void setTransitionDipoleGradient(const std::vector<Eigen::MatrixXd> &);

  int getNumberOfAtoms() const;
  Eigen::MatrixXd getGroundStateEnergyGradient() const;
  Eigen::MatrixXd getExcitedStateEnergyGradient() const;
  std::vector<Eigen::MatrixXd> getCenterOfMassGradient() const;
  std::vector<Eigen::MatrixXd> getGroundStateDipoleGradient() const;
  std::vector<Eigen::MatrixXd> getExcitedStateDipoleGradient() const;
  std::vector<Eigen::MatrixXd> getTransitionDipoleGradient() const;

  Eigen::Vector3d getDipoleSum() const;
  Eigen::Vector3d getDipoleDifference() const;
  Eigen::Vector3d getGroundStateDipole() const;
  Eigen::Vector3d getExcitedStateDipole() const;
  Eigen::Vector3d getTransitionDipole() const;
  Eigen::Vector3d getCenterOfMass() const;

  double getX(Monomer &);
  double getZ(Monomer &);
  double getXX(Monomer &);
  double getXZ(Monomer &);
  double getZX(Monomer &);
  double getZZ(Monomer &);
};
