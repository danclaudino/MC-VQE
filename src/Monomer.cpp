#include "Monomer.hpp"

double Monomer::twoBodyH(const Eigen::VectorXd &mu_A,
                         const Eigen::VectorXd &mu_B,
                         const Eigen::VectorXd &r_AB) {
  /** Lambda function to compute the two-body AIEM Hamiltonian matrix elements
   * @param[in] mu_A Some dipole moment vector (gs/es/t) from chromophore A
   * @param[in] mu_B Some dipole moment vector (gs/es/t) from chromophore B
   * @param[in] r_AB Vector between A and B
   */
  auto d_AB = r_AB.norm(); // Distance between A and B
  auto n_AB = r_AB / d_AB; // Normal vector along r_AB
  return (mu_A.dot(mu_B) - 3.0 * mu_A.dot(n_AB) * mu_B.dot(n_AB)) /
         std::pow(d_AB, 3.0); // return matrix element
}

void Monomer::isDipoleInDebye(const bool isDebye) {

  if (isDebye) {
    _groundStateDipole *= DEBYE2AU;
    _excitedStateDipole *= DEBYE2AU;
    return;
  } else {
    return;
  }
}

void Monomer::isCenterOfMassInAngstrom(const bool isAngstrom) {
  if (isAngstrom) {
    _centerOfMass *= ANGSTROM2BOHR;
    return;
  } else {
    return;
  }
}

void Monomer::setNumberOfAtoms(const int nAtoms) { _nAtoms = nAtoms; }

void Monomer::setGroundStateEnergy(const double groundStateEnergy) {
  _groundStateEnergy = groundStateEnergy;
}

void Monomer::setExcitedStateEnergy(const double excitedStateEnergy) {
  _excitedStateEnergy = excitedStateEnergy;
}

void Monomer::setGroundStateDipole(const Eigen::Vector3d groundStateDipole) {
  _groundStateDipole = groundStateDipole;
}

void Monomer::setExcitedStateDipole(const Eigen::Vector3d excitedStateDipole) {
  _excitedStateDipole = excitedStateDipole;
}

void Monomer::setTransitionDipole(const Eigen::Vector3d transitionDipole) {
  _transitionDipole = transitionDipole;
}

void Monomer::setCenterOfMass(const Eigen::Vector3d centerOfMass) {
  _centerOfMass = centerOfMass;
}

void Monomer::setGroundStateEnergyGradient(
    const Eigen::MatrixXd &groundStateEnergyGradient) {
  _groundStateEnergyGradient = groundStateEnergyGradient;
}

void Monomer::setExcitedStateEnergyGradient(
    const Eigen::MatrixXd &excitedStateEnergyGradient) {
  _excitedStateEnergyGradient = excitedStateEnergyGradient;
}

void Monomer::setGroundStateDipoleGradient(
    const std::vector<Eigen::MatrixXd> &groundStateDipoleGradient) {
  _groundStateDipoleGradient = groundStateDipoleGradient;
}

void Monomer::setExcitedStateDipoleGradient(
    const std::vector<Eigen::MatrixXd> &excitedStateDipoleGradient) {
  _excitedStateDipoleGradient = excitedStateDipoleGradient;
}

void Monomer::setTransitionDipoleGradient(
    const std::vector<Eigen::MatrixXd> &transitionDipoleGradient) {
  _transitionDipoleGradient = transitionDipoleGradient;
}

void Monomer::setCenterOfMassGradient(
    const std::vector<Eigen::MatrixXd> &centerOfMassGradient) {
  _centerOfMassGradient = centerOfMassGradient;
}

int Monomer::getNumberOfAtoms() const { return _nAtoms; }

Eigen::MatrixXd Monomer::getGroundStateEnergyGradient() const {
  return _groundStateEnergyGradient;
}

Eigen::MatrixXd Monomer::getExcitedStateEnergyGradient() const {
  return _excitedStateEnergyGradient;
}

std::vector<Eigen::MatrixXd> Monomer::getGroundStateDipoleGradient() const {
  return _groundStateDipoleGradient;
}

std::vector<Eigen::MatrixXd> Monomer::getExcitedStateDipoleGradient() const {
  return _excitedStateDipoleGradient;
}

std::vector<Eigen::MatrixXd> Monomer::getTransitionDipoleGradient() const {
  return _transitionDipoleGradient;
}

std::vector<Eigen::MatrixXd> Monomer::getCenterOfMassGradient() const {
  return _centerOfMassGradient;
}

double Monomer::getS() const {
  return 0.5 * (_groundStateEnergy + _excitedStateEnergy);
}
double Monomer::getD() const {
  return 0.5 * (_groundStateEnergy - _excitedStateEnergy);
}
double Monomer::getE() const { return _excitedStateEnergy; }

Eigen::Vector3d Monomer::getDipoleSum() const {
  return 0.5 * (_groundStateDipole + _excitedStateDipole);
}
Eigen::Vector3d Monomer::getDipoleDifference() const {
  return 0.5 * (_groundStateDipole - _excitedStateDipole);
}
Eigen::Vector3d Monomer::getGroundStateDipole() const {
  return _groundStateDipole;
}
Eigen::Vector3d Monomer::getExcitedStateDipole() const {
  return _excitedStateDipole;
}
Eigen::Vector3d Monomer::getTransitionDipole() const {
  return _transitionDipole;
}
Eigen::Vector3d Monomer::getCenterOfMass() const { return _centerOfMass; }

double Monomer::getX(Monomer &B) {
  auto r = _centerOfMass - B.getCenterOfMass();
  return 0.5 * (twoBodyH(_transitionDipole, B.getDipoleSum(), r) +
                twoBodyH(B.getDipoleSum(), _transitionDipole, r));
}

double Monomer::getZ(Monomer &B) {
  auto r = _centerOfMass - B.getCenterOfMass();
  return 0.5 * (twoBodyH(getDipoleDifference(), B.getDipoleSum(), r) +
                twoBodyH(B.getDipoleSum(), getDipoleDifference(), r));
}

double Monomer::getXX(Monomer &B) {
  auto r = _centerOfMass - B.getCenterOfMass();
  return 0.25 * (twoBodyH(_transitionDipole, B.getTransitionDipole(), r) +
                 twoBodyH(B.getTransitionDipole(), _transitionDipole, r));
}

double Monomer::getXZ(Monomer &B) {
  auto r = _centerOfMass - B.getCenterOfMass();
  return 0.25 * (twoBodyH(_transitionDipole, B.getDipoleDifference(), r) +
                 twoBodyH(B.getDipoleDifference(), _transitionDipole, r));
}

double Monomer::getZX(Monomer &B) {
  auto r = _centerOfMass - B.getCenterOfMass();
  return 0.25 * (twoBodyH(getDipoleDifference(), B.getTransitionDipole(), r) +
                 twoBodyH(B.getTransitionDipole(), getDipoleDifference(), r));
}

double Monomer::getZZ(Monomer &B) {
  auto r = _centerOfMass - B.getCenterOfMass();
  return 0.25 * (twoBodyH(getDipoleDifference(), B.getDipoleDifference(), r) +
                 twoBodyH(B.getDipoleDifference(), getDipoleDifference(), r));
}
