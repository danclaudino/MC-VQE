#include "mc-vqe.hpp"

Monomer::Monomer(const double groundStateEnergy,
                 const double excitedStateEnergy,
                 const Eigen::Vector3d groundStateDipole,
                 const Eigen::Vector3d excitedStateDipole,
                 const Eigen::Vector3d transitionDipole,
                 const Eigen::Vector3d centerOfMass) {
  _groundStateEnergy = groundStateEnergy;
  _excitedStateEnergy = excitedStateEnergy;
  _groundStateDipole = groundStateDipole;
  _excitedStateDipole = excitedStateDipole;
  _transitionDipole = transitionDipole;
  _centerOfMass = centerOfMass;
}

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

double Monomer::getS() {
  return 0.5 * (_groundStateEnergy + _excitedStateEnergy);
}
double Monomer::getD() {
  return 0.5 * (_groundStateEnergy - _excitedStateEnergy);
}
double Monomer::getE() { return _excitedStateEnergy; }

Eigen::Vector3d Monomer::getDipoleSum() {
  return 0.5 * (_groundStateDipole + _excitedStateDipole);
}
Eigen::Vector3d Monomer::getDipoleDifference() {
  return 0.5 * (_groundStateDipole - _excitedStateDipole);
}
Eigen::Vector3d Monomer::getGroundStateDipole() { return _groundStateDipole; }
Eigen::Vector3d Monomer::getExcitedStateDipole() { return _excitedStateDipole; }
Eigen::Vector3d Monomer::getTransitionDipole() { return _transitionDipole; }
Eigen::Vector3d Monomer::getCenterOfMass() { return _centerOfMass; }

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

