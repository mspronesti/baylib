#pragma once

#include <map>
#include <string>

#include "../graphs/graphStructure/GraphElements.h"

class VariableNodeMap {
public:
	VariableNodeMap();

	VariableNodeMap(const VariableNodeMap& vnm);

	~VariableNodeMap();

	int addVariable(const std::string& variableName);

	bool checkIfExists(const std::string& variableName);

	int getNumberOfVariables();

	void removeVariable(const std::string& variableName);

	NodeId idFromName(const std::string& variableName);

private:
	NodeId m_lastId;
	std::map<std::string, NodeId> m_variableNodeMap;
};