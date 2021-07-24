#include "VariableNodeMap.h"

VariableNodeMap::VariableNodeMap() : m_lastId(0) {

}

VariableNodeMap::VariableNodeMap(const VariableNodeMap& vnm) {
	m_variableNodeMap.clear();
	m_variableNodeMap = vnm.m_variableNodeMap;
}

VariableNodeMap::~VariableNodeMap() {

}

int VariableNodeMap::addVariable(const std::string& variableName) {
	if (!checkIfExists(variableName)) {
		m_variableNodeMap.emplace(variableName, m_lastId);
		return m_lastId++;
	}
}

bool VariableNodeMap::checkIfExists(const std::string& variableName) {
	if (m_variableNodeMap.find(variableName) != m_variableNodeMap.end()) return true;
	else return false;
}

int VariableNodeMap::getNumberOfVariables() {
	return m_lastId;
}

void VariableNodeMap::removeVariable(const std::string& variableName) {
	if (checkIfExists(variableName))
		m_variableNodeMap.erase(variableName);
}

//returns the id associated to the variable.
//if the variable does not exist, id -1 is returned
NodeId VariableNodeMap::idFromName(const std::string& variableName) {
	auto it = m_variableNodeMap.find(variableName);
	if (it != m_variableNodeMap.end())
		return it->second;
	else return -1;
}