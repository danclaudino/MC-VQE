#include "mc-vqe.hpp"

using namespace xacc;

namespace xacc {
namespace algorithm {

std::shared_ptr<CompositeInstruction>
MC_VQE::statePreparationCircuit(const Eigen::VectorXd &angles) const {
  /** Constructs the circuit that prepares a CIS state
   * @param[in] angles Angles to parameterize CIS state
   */

  // Provider to create IR for CompositeInstruction, aka circuit
  auto provider = xacc::getIRProvider("quantum");
  // allocate memory for circuit
  auto cisCircuit = provider->createComposite("cisCircuit");

  // Beginning of CIS state preparation
  //
  // Ry "pump"
  auto ry = provider->createInstruction("Ry", {0}, {angles(0)});
  cisCircuit->addInstruction(ry);

  // Fy gates = Ry(-theta/2)-CZ-Ry(theta/2) = Ry(-theta/2)-H-CNOT-H-Ry(theta/2)
  //
  // |A>------------------------o------------------------
  //                            |
  // |B>--[Ry(-angles(i)/2)]-H-[X]-H-[Ry(+angles(i)/2)]--
  //
  for (int i = 1; i < nChromophores; i++) {

    std::size_t control = i - 1, target = i;
    // Ry(-theta/2)
    auto ry_minus =
        provider->createInstruction("Ry", {target}, {-angles(2 * i - 1) / 2});
    cisCircuit->addInstruction(ry_minus);

    auto hadamard1 = provider->createInstruction("H", {target});
    cisCircuit->addInstruction(hadamard1);

    auto cnot = provider->createInstruction("CNOT", {control, target});
    cisCircuit->addInstruction(cnot);

    auto hadamard2 = provider->createInstruction("H", {target});
    cisCircuit->addInstruction(hadamard2);

    // Ry(+theta/2)
    auto ry_plus =
        provider->createInstruction("Ry", {target}, {angles(2 * i) / 2});
    cisCircuit->addInstruction(ry_plus);
  }

  // Wall of CNOTs
  for (int i = nChromophores - 2; i >= 0; i--) {
    for (int j = nChromophores - 1; j > i; j--) {

      std::size_t control = j, target = i;
      auto cnot = provider->createInstruction("CNOT", {control, target});
      cisCircuit->addInstruction(cnot);
    }
  }
  // end of CIS state preparation
  return cisCircuit;
}

std::shared_ptr<CompositeInstruction> MC_VQE::entanglerCircuit() const {
  /** Constructs the entangler part of the circuit
   */

  // Create CompositeInstruction for entangler
  auto provider = xacc::getIRProvider("quantum");
  auto entanglerCircuit = provider->createComposite("entanglerCircuit");

  // structure of entangler
  // does not implement the first two Ry to remove redundancies in the circuit
  // and thus reducing the number of variational parameters
  //
  // |A>--[Ry(x0)]--o--[Ry(x2)]--o--[Ry(x4)]-
  //                |            |
  // |B>--[Ry(x1)]--x--[Ry(x3)]--x--[Ry(x5)]-
  //
  auto entanglerGate = [&](const std::size_t control, const std::size_t target,
                           int paramCounter) {
    /** Lambda function to construct the entangler gates
     * Interleaves CNOT(control, target) and Ry rotations
     *
     * @param[in] control Index of the control/source qubit
     * @param[in] target Index of the target qubit
     * @param[in] paramCounter Variable that keeps track of and indexes the
     * circuit parameters
     */

    auto cnot1 = provider->createInstruction("CNOT", {control, target});
    entanglerCircuit->addInstruction(cnot1);

    auto ry1 = provider->createInstruction(
        "Ry", {control},
        {InstructionParameter("x" + std::to_string(paramCounter))});
    entanglerCircuit->addVariable("x" + std::to_string(paramCounter));
    entanglerCircuit->addInstruction(ry1);

    auto ry2 = provider->createInstruction(
        "Ry", {target},
        {InstructionParameter("x" + std::to_string(paramCounter + 1))});
    entanglerCircuit->addVariable("x" + std::to_string(paramCounter + 1));
    entanglerCircuit->addInstruction(ry2);

    auto cnot2 = provider->createInstruction("CNOT", {control, target});
    entanglerCircuit->addInstruction(cnot2);

    auto ry3 = provider->createInstruction(
        "Ry", {control},
        {InstructionParameter("x" + std::to_string(paramCounter + 2))});
    entanglerCircuit->addVariable("x" + std::to_string(paramCounter + 2));
    entanglerCircuit->addInstruction(ry3);

    auto ry4 = provider->createInstruction(
        "Ry", {target},
        {InstructionParameter("x" + std::to_string(paramCounter + 3))});
    entanglerCircuit->addVariable("x" + std::to_string(paramCounter + 3));
    entanglerCircuit->addInstruction(ry4);
  };

  // placing the first Ry's in the circuit
  int paramCounter = 0;
  for (int i = 0; i < nChromophores; i++) {
    // std::size_t q = i;
    auto ry = provider->createInstruction(
        "Ry", {(std::size_t)i},
        {InstructionParameter("x" + std::to_string(paramCounter))});
    entanglerCircuit->addVariable("x" + std::to_string(paramCounter));
    entanglerCircuit->addInstruction(ry);

    paramCounter++;
  }

  // placing the entanglerGates in the circuit
  for (int layer : {0, 1}) {
    for (int i = layer; i < nChromophores - layer; i += 2) {
      std::size_t control = i, target = i + 1;
      entanglerGate(control, target, paramCounter);
      paramCounter += NPARAMSENTANGLER;
    }
  }

  // if the molecular system is cyclic, we need to add this last entangler
  if (isCyclic) {
    std::size_t control = nChromophores - 1, target = 0;
    entanglerGate(control, target, paramCounter);
  }

  // end of entangler circuit construction
  return entanglerCircuit;
}

} // namespace algorithm
} // namespace xacc