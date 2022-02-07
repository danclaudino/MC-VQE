#include "Circuit.hpp"
#include "mc-vqe.hpp"
#include <Eigen/Eigenvalues>
#include <complex>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace xacc;
using namespace xacc::quantum;

namespace xacc {
namespace algorithm {

/**
 * @brief Return shared_ptr to CIS state preparation circuit
 * 
 * @param angles State preparation angles
 * @return CIS state preparation circuit 
 */
std::shared_ptr<CompositeInstruction>
MC_VQE::statePreparationCircuit(const Eigen::VectorXd &angles) const {

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

/**
 * @brief Return shared_ptr to entangler
 * 
 * @return Entangler
 */
std::shared_ptr<CompositeInstruction>
MC_VQE::entanglerCircuit() const {
  
  // Create CompositeInstruction for entangler
  auto provider = xacc::getIRProvider("quantum");
  auto entanglerCircuit = provider->createComposite("entanglerCircuit");
  int varIdx = 0;

 if (entanglerType == "trotterized") {

    auto entangler = [&](const std::size_t control, const std::size_t target) {

      std::vector<std::shared_ptr<Instruction>> gates;
      std::vector<std::string> varNames;

      auto varName = "x" + std::to_string(varIdx++);
      auto ry = provider->createInstruction("Ry", {target}, {varName});
      varNames.push_back(varName);
      gates.push_back(ry);

      varName = "x" + std::to_string(varIdx++);
      ry = provider->createInstruction("Ry", {target}, {varName});
      varNames.push_back(varName);
      gates.push_back(ry);

      auto rx2 = provider->createInstruction("U", {control}, {PI / 2.0, 3.0 * PI / 2.0, PI / 2.0});
      gates.push_back(rx2);

      auto h = provider->createInstruction("H", {target});
      gates.push_back(h);

      auto cnot = provider->createInstruction("CNOT", {control, target});
      gates.push_back(cnot);

      varName = "x" + std::to_string(varIdx++);
      auto rz = provider->createInstruction("Rz", {target}, {varName});
      varNames.push_back(varName);
      gates.push_back(rz);

      cnot = provider->createInstruction("CNOT", {control, target});
      gates.push_back(cnot);

      h = provider->createInstruction("H", {target});
      gates.push_back(h);

      cnot = provider->createInstruction("CNOT", {control, target});
      gates.push_back(cnot);

      varName = "x" + std::to_string(varIdx++);
      rz = provider->createInstruction("Rz", {target}, {varName});
      varNames.push_back(varName);
      gates.push_back(rz);

      cnot = provider->createInstruction("CNOT", {control, target});
      gates.push_back(cnot);

      auto rx2T = provider->createInstruction("U", {control}, {PI / 2.0, PI / 2.0, 3 * PI / 2.0});
      gates.push_back(rx2T);

      rx2 = provider->createInstruction("U", {target}, {PI / 2.0, 3.0 * PI / 2.0, PI / 2.0});
      gates.push_back(rx2);

      cnot = provider->createInstruction("CNOT", {control, target});
      gates.push_back(cnot);

      varName = "x" + std::to_string(varIdx++);
      rz = provider->createInstruction("Rz", {target}, {varName});
      varNames.push_back(varName);
      gates.push_back(rz);

      cnot = provider->createInstruction("CNOT", {control, target});
      gates.push_back(cnot);

      h = provider->createInstruction("H", {control});
      gates.push_back(h);

      cnot = provider->createInstruction("CNOT", {control, target});
      gates.push_back(cnot);

      varName = "x" + std::to_string(varIdx++);
      rz = provider->createInstruction("Rz", {target}, {varName});
      varNames.push_back(varName);
      gates.push_back(rz);

      cnot = provider->createInstruction("CNOT", {control, target});
      gates.push_back(cnot);

      h = provider->createInstruction("H", {control});
      gates.push_back(h);

      rx2T = provider->createInstruction("U", {target}, {PI / 2.0, PI / 2.0, 3.0 * PI / 2.0});
      gates.push_back(rx2T);

      entanglerCircuit->addVariables(varNames);
      entanglerCircuit->addInstructions(gates);
    };

    // placing the entanglerGates in the circuit
    for (int layer : {0, 1}) {
      for (int i = layer; i < nChromophores - layer; i += 2) {
        std::size_t control = i, target = i + 1;
        entangler(control, target);
      }
    }

    // if the molecular system is cyclic, we need to add this last entangler
    if (isCyclic) {
      std::size_t control = nChromophores - 1, target = 0;
      entangler(control, target);
    } 

  } 
  
  if (entanglerType == "default") {

    auto entangler = [&](const std::size_t control, const std::size_t target) {

      std::vector<std::shared_ptr<Instruction>> gates;
      std::vector<std::string> varNames;
      auto cnot = provider->createInstruction("CNOT", {control, target});
      gates.push_back(cnot);

      std::string varName = "x" + std::to_string(varIdx++);
      auto ry = provider->createInstruction("Ry", {target}, {varName});
      gates.push_back(ry);
      varNames.push_back(varName);

      varName = "x" + std::to_string(varIdx++);
      ry = provider->createInstruction("Ry", {control}, {varName});
      gates.push_back(ry);
      varNames.push_back(varName);

      cnot = provider->createInstruction("CNOT", {control, target});
      gates.push_back(cnot);

      varName = "x" + std::to_string(varIdx++);
      ry = provider->createInstruction("Ry", {target}, {varName});
      gates.push_back(ry);
      varNames.push_back(varName);

      varName = "x" + std::to_string(varIdx++);
      ry = provider->createInstruction("Ry", {control},{varName});
      gates.push_back(ry);
      varNames.push_back(varName);

      entanglerCircuit->addVariables(varNames);
      entanglerCircuit->addInstructions(gates);
    };

    // placing the first Ry's in the circuit
    for (std::size_t i = 0; i < nChromophores; i++) {
      std::string varName = "x" + std::to_string(varIdx++);
      auto ry = provider->createInstruction("Ry", {i}, {varName});
      entanglerCircuit->addVariable(varName);
      entanglerCircuit->addInstruction(ry);
    }

    // placing the entanglerGates in the circuit
    for (int layer : {0, 1}) {
      for (int i = layer; i < nChromophores - layer; i += 2) {
        std::size_t control = i, target = i + 1;
        entangler(control, target);
      }
    }

    // if the molecular system is cyclic, we need to add this last entangler
    if (isCyclic) {
      std::size_t control = nChromophores - 1, target = 0;
      entangler(control, target);
    }

  }

  if (entanglerType == "Ry") {
    for (std::size_t i = 0; i < nChromophores; i++) {
      std::string varName = "x" + std::to_string(varIdx++);
      auto ry = provider->createInstruction("Ry", {i}, {varName});
      entanglerCircuit->addVariable(varName);
      entanglerCircuit->addInstruction(ry);
    }
  }

  return entanglerCircuit;
}

} // namespace algorithm
} // namespace xacc