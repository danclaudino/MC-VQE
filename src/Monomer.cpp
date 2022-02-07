#include "Monomer.hpp"

/**
 * @brief Construct a new Monomer:: Monomer object
 *
 * @param groundStateEnergy ground state energy
 * @param excitedStateEnergy excited state energy
 * @param groundStateDipole ground state dipole moment
 * @param excitedStateDipole excited state dipole moment
 * @param transitionDipole transition dipole moment <1|u|0>
 * @param centerOfMass center of mass/centroid
 */
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

/**
 * @brief
 *
 * @param mu_A Some dipole moment vector (gs/es/t) from monomer A
 * @param mu_B Some dipole moment vector (gs/es/t) from monomer B
 * @param r_AB Vector between A and B
 * @return double 2-body Hamiltonian term
 */
double Monomer::twoBodyH(const Eigen::VectorXd &mu_A,
                         const Eigen::VectorXd &mu_B,
                         const Eigen::VectorXd &r_AB) const {

  auto d_AB = r_AB.norm(); // Distance between A and B
  auto n_AB = r_AB / d_AB; // Normal vector along r_AB
  return (mu_A.dot(mu_B) - 3.0 * mu_A.dot(n_AB) * mu_B.dot(n_AB)) /
         std::pow(d_AB, 3.0); // return matrix element
}

/**
 * @brief Convert the ground and excited state dipole moment to Debye
 *
 * @param isDebye Boolean that controls whether to convert to Debye
 */
void Monomer::isDipoleInDebye(const bool isDebye) {

  if (isDebye) {
    _groundStateDipole *= DEBYE2AU;
    _excitedStateDipole *= DEBYE2AU;
    return;
  } else {
    return;
  }
}

/**
 * @brief Convert the center of mass to Angstrom
 *
 * @param isAngstrom Boolean that controls whether to convert to Angstrom
 */
void Monomer::isCenterOfMassInAngstrom(const bool isAngstrom) {
  if (isAngstrom) {
    _centerOfMass *= ANGSTROM2BOHR;
    return;
  } else {
    return;
  }
}

/**
 * @brief Set the number of atoms in the monomer
 *
 * @param nAtoms Number of atoms
 */
void Monomer::setNumberOfAtoms(const int nAtoms) { _nAtoms = nAtoms; }

/**
 * @brief Set the ground state energy
 *
 * @param groundStateEnergy Ground state energy
 */
void Monomer::setGroundStateEnergy(const double groundStateEnergy) {
  _groundStateEnergy = groundStateEnergy;
}

/**
 * @brief Set the excited state energy
 *
 * @param excitedStateEnergy Excited state energy
 */
void Monomer::setExcitedStateEnergy(const double excitedStateEnergy) {
  _excitedStateEnergy = excitedStateEnergy;
}

/**
 * @brief Set the ground state dipole moment
 *
 * @param groundStateDipole Ground state dipole moment
 */
void Monomer::setGroundStateDipole(const Eigen::Vector3d groundStateDipole) {
  _groundStateDipole = groundStateDipole;
}

/**
 * @brief Set the excited state dipole moment
 *
 * @param excitedStateDipole Excited state dipole moment
 */
void Monomer::setExcitedStateDipole(const Eigen::Vector3d excitedStateDipole) {
  _excitedStateDipole = excitedStateDipole;
}

/**
 * @brief Set the transition dipole moment
 *
 * @param transitionDipole Transition dipole moment
 */
void Monomer::setTransitionDipole(const Eigen::Vector3d transitionDipole) {
  _transitionDipole = transitionDipole;
}

/**
 * @brief Set the center of mass/centroid
 *
 * @param centerOfMass Center of mass
 */
void Monomer::setCenterOfMass(const Eigen::Vector3d centerOfMass) {
  _centerOfMass = centerOfMass;
}

/**
 * @brief Set the ground state energy gradient
 *
 * @param groundStateEnergyGradient Ground state energy gradient
 */
void Monomer::setGroundStateEnergyGradient(
    const Eigen::MatrixXd &groundStateEnergyGradient) {
  _groundStateEnergyGradient = groundStateEnergyGradient;
}

/**
 * @brief Set the excited state energy gradient
 *
 * @param excitedStateEnergyGradient Excited state energy gradient
 */
void Monomer::setExcitedStateEnergyGradient(
    const Eigen::MatrixXd &excitedStateEnergyGradient) {
  _excitedStateEnergyGradient = excitedStateEnergyGradient;
}

/**
 * @brief Set the ground state dipole moment gradient
 *
 * @param groundStateDipoleGradient Ground state dipole moment gradient
 */
void Monomer::setGroundStateDipoleGradient(
    const std::vector<Eigen::MatrixXd> &groundStateDipoleGradient) {
  _groundStateDipoleGradient = groundStateDipoleGradient;
}

/**
 * @brief Set the excited state dipole moment gradient
 *
 * @param excitedStateDipoleGradient Excited state dipole moment gradient
 */
void Monomer::setExcitedStateDipoleGradient(
    const std::vector<Eigen::MatrixXd> &excitedStateDipoleGradient) {
  _excitedStateDipoleGradient = excitedStateDipoleGradient;
}

/**
 * @brief Set the transition dipole moment gradient
 *
 * @param transitionDipoleGradient Transition dipole moment gradient
 */
void Monomer::setTransitionDipoleGradient(
    const std::vector<Eigen::MatrixXd> &transitionDipoleGradient) {
  _transitionDipoleGradient = transitionDipoleGradient;
}

/**
 * @brief Set the center of mass gradient
 *
 * @param centerOfMassGradient Center of mass gradient
 */
void Monomer::setCenterOfMassGradient(
    const std::vector<Eigen::MatrixXd> &centerOfMassGradient) {
  _centerOfMassGradient = centerOfMassGradient;
}

/**
 * @brief Return the number of atoms
 *
 * @return Number of atoms
 */
int Monomer::getNumberOfAtoms() const { return _nAtoms; }

/**
 * @brief Return the ground state energy gradient
 *
 * @return Ground state energy gradient
 */
Eigen::MatrixXd Monomer::getGroundStateEnergyGradient() const {
  return _groundStateEnergyGradient;
}

/**
 * @brief Return the excited state energy gradient
 *
 * @return Excited state energy gradient
 */
Eigen::MatrixXd Monomer::getExcitedStateEnergyGradient() const {
  return _excitedStateEnergyGradient;
}

/**
 * @brief Return the ground state dipole moment gradient
 *
 * @return Ground state dipole moment gradient
 */
std::vector<Eigen::MatrixXd> Monomer::getGroundStateDipoleGradient() const {
  return _groundStateDipoleGradient;
}

/**
 * @brief Return the excited state dipole moment gradient
 *
 * @return Excited state dipole moment gradient
 */
std::vector<Eigen::MatrixXd> Monomer::getExcitedStateDipoleGradient() const {
  return _excitedStateDipoleGradient;
}

/**
 * @brief Return the transition dipole moment gradient
 *
 * @return Transition dipole moment gradient
 */
std::vector<Eigen::MatrixXd> Monomer::getTransitionDipoleGradient() const {
  return _transitionDipoleGradient;
}

/**
 * @brief Return the center of mass gradient
 *
 * @return Center of mass gradient
 */
std::vector<Eigen::MatrixXd> Monomer::getCenterOfMassGradient() const {
  return _centerOfMassGradient;
}

/**
 * @brief Return the sum of ground and excited state energies
 *
 * @return Sum of ground and excited state energies
 */
double Monomer::getS() const {
  return 0.5 * (_groundStateEnergy + _excitedStateEnergy);
}

/**
 * @brief Return the difference of ground and excited state energies
 *
 * @return Difference of ground and excited state energies
 */
double Monomer::getD() const {
  return 0.5 * (_groundStateEnergy - _excitedStateEnergy);
}

/**
 * @brief Return the excited state energy
 *
 * @return Excited state energy
 */
double Monomer::getE() const { return _excitedStateEnergy; }

/**
 * @brief Return the sum of the ground and excited state dipole moments
 *
 * @return Sum of the ground and excited state dipole moments
 */
Eigen::Vector3d Monomer::getDipoleSum() const {
  return 0.5 * (_groundStateDipole + _excitedStateDipole);
}

/**
 * @brief Return the difference of the ground and excited state dipole moments
 *
 * @return Difference of the ground and excited state dipole moments
 */
Eigen::Vector3d Monomer::getDipoleDifference() const {
  return 0.5 * (_groundStateDipole - _excitedStateDipole);
}

/**
 * @brief Return the ground state dipole moment
 *
 * @return Ground state dipole moment
 */
Eigen::Vector3d Monomer::getGroundStateDipole() const {
  return _groundStateDipole;
}

/**
 * @brief Return the excited state dipole moment
 *
 * @return Excited state dipole moment
 */
Eigen::Vector3d Monomer::getExcitedStateDipole() const {
  return _excitedStateDipole;
}

/**
 * @brief Return the transition dipole moment
 *
 * @return Transition dipole moment
 */
Eigen::Vector3d Monomer::getTransitionDipole() const {
  return _transitionDipole;
}

/**
 * @brief Return the center of mass
 *
 * @return Center of mass
 */
Eigen::Vector3d Monomer::getCenterOfMass() const { return _centerOfMass; }

/**
 * @brief Return the X_A coefficient in the Pauli basis Hamiltonian (Eq. 35)
 *
 * @param B Monomer B
 * @return X_A
 */
double Monomer::getX(const Monomer &B) const {
  auto r = _centerOfMass - B.getCenterOfMass();
  return 0.5 * (twoBodyH(_transitionDipole, B.getDipoleSum(), r) +
                twoBodyH(B.getDipoleSum(), _transitionDipole, r));
}

/**
 * @brief Return the Z_A coefficient in the Pauli basis Hamiltonian (Eq. 36)
 *
 * @param B Monomer B
 * @return Z_A
 */
double Monomer::getZ(const Monomer &B) const {
  auto r = _centerOfMass - B.getCenterOfMass();
  return 0.5 * (twoBodyH(getDipoleDifference(), B.getDipoleSum(), r) +
                twoBodyH(B.getDipoleSum(), getDipoleDifference(), r));
}

/**
 * @brief Return the X_AX_B coefficient in the Pauli basis Hamiltonian (Eq. 37)
 *
 * @param B Monomer B
 * @return X_AX_B
 */
double Monomer::getXX(const Monomer &B) const {
  auto r = _centerOfMass - B.getCenterOfMass();
  return 0.25 * (twoBodyH(_transitionDipole, B.getTransitionDipole(), r) +
                 twoBodyH(B.getTransitionDipole(), _transitionDipole, r));
}

/**
 * @brief Return the X_AZ_B coefficient in the Pauli basis Hamiltonian (Eq. 38)
 *
 * @param B Monomer B
 * @return X_AZ_B
 */
double Monomer::getXZ(const Monomer &B) const {
  auto r = _centerOfMass - B.getCenterOfMass();
  return 0.25 * (twoBodyH(_transitionDipole, B.getDipoleDifference(), r) +
                 twoBodyH(B.getDipoleDifference(), _transitionDipole, r));
}

/**
 * @brief Return the Z_AX_B coefficient in the Pauli basis Hamiltonian (Eq. 39)
 *
 * @param B Monomer B
 * @return Z_AX_B
 */
double Monomer::getZX(const Monomer &B) const {
  auto r = _centerOfMass - B.getCenterOfMass();
  return 0.25 * (twoBodyH(getDipoleDifference(), B.getTransitionDipole(), r) +
                 twoBodyH(B.getTransitionDipole(), getDipoleDifference(), r));
}

/**
 * @brief Return the Z_AZ_B coefficient in the Pauli basis Hamiltonian (Eq. 40)
 *
 * @param B Monomer B
 * @return Z_AZ_B
 */
double Monomer::getZZ(const Monomer &B) const {
  auto r = _centerOfMass - B.getCenterOfMass();
  return 0.25 * (twoBodyH(getDipoleDifference(), B.getDipoleDifference(), r) +
                 twoBodyH(B.getDipoleDifference(), getDipoleDifference(), r));
}
