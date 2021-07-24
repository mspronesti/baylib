#pragma once

#include <string>
#include <fstream>
#include <sstream>

#include "../../parser/rapidxml/rapidxml.hpp"
#include "../BayesianNet.h"
#include "../../probability/CPT.h"

using namespace rapidxml;

//reader class for bayesian network stored in .xdsl files
//the template parameter is used to define the precision of the probability red from the file

template <class T = float>
class BNReader {
public:
	BNReader() {};

	//loads the bayesian network from the given file
	void loadNetworkFromFile(const std::string& fileName, std::shared_ptr<BayesianNetwork<T>> bn) {
		auto doc = std::make_shared<xml_document<>>();
		std::ifstream inputFile(fileName);
		auto buffer = std::make_shared<std::stringstream>();
		*buffer << inputFile.rdbuf();
		inputFile.close();
		std::string content(buffer->str());
		doc->parse<0>(&content[0]);

		xml_node<>* pRoot = doc->first_node();

		xml_node<>* pNodes = pRoot->first_node("nodes");

		//reading all variables in file
		for (xml_node<>* pNode = pNodes->first_node("cpt"); pNode; pNode = pNode->next_sibling()) {

			xml_attribute<>* attr = pNode->first_attribute("id");

			std::string varName;

			std::vector<std::string> stateNames;

			int nStates;

			std::string parents;

			std::vector<VarStates> variablesOrder;

			std::vector<std::vector<int>> variablesCombinations;

			std::string probDistribution;

			std::string resultingStates;

			std::vector<std::string> resultingStatesSplitted;

			//reading variable name
			varName.append(attr->value());

			//reading all properties of the variable
			for (xml_node<>* pStates = pNode->first_node("state"); pStates; pStates = pStates->next_sibling()) {
				std::string name(pStates->name());

				//reading variable's states
				if (name == "state") {
					attr = pStates->first_attribute("id");
					if (attr != NULL)
						stateNames.emplace_back(attr->value());
				}
				else if (name == "resultingstates") {
					resultingStates.append(pStates->value());
				}
				//reading cpt of the variable
				else if (name == "probabilities") {
					probDistribution.append(pStates->value());
				}
				//reading the parents order of the variable
				else if (name == "parents") {
					parents.append(pStates->value());
				}
				//else if (name == "property") {
				//	attr = pStates->first_attribute("id");
				//	std::string id(attr->value());


					//if (id == "VID") {
					//	varName.append(pStates->value());
					//}

					//else if (id == "parents_order") {
					//	parents.append(pStates->value());
					//}

					//else if (id == "cpt") {
					//	probDistribution.append(pStates->value());

					//}
				//}

			}

			nStates = stateNames.size();

			splitParents(parents, variablesOrder, bn);

			//the conditioned variable of the CPT is added last in the variables order
			variablesOrder.emplace_back(bn->getNextVariableId(), nStates);

			int numResultingStates = splitResultingStates(resultingStates, resultingStatesSplitted);

			CPT<T> cpt(stateNames);

			cpt.addVariablesOrder(std::move(variablesOrder));

			splitProbabilities(probDistribution, variablesOrder, cpt);

			if (numResultingStates > 0)
				cpt.addResultingStates(std::move(resultingStatesSplitted));

			bn->addVariable(varName, cpt);

		}

		bn->checkSparseCPTs();

		bn->addArcsFromCPTs();
	}

private:

	//given the vector of variables for the current CPT, inserts the initial combination into combination
	void getInitialCombination(std::vector<VarStates>& states, std::vector<int>& combination) {
		for (int i = 0; i < states.size(); i++) {
			combination.push_back(0);
		}
	}

	//given the vector of variables for the current CPT, inserts in combination the next one to the that already stored
	void getNextCombination(std::vector<VarStates>& states, std::vector<int>& combination) {
		int inc = 1;

		for (int i = combination.size() - 1; i > -1; i--) {
			combination[i] += inc;
			combination[i] = combination[i] % states[i].m_nStates;
			if (combination[i] != 0) break;
		}
	}

	//returns the number of combinations of states for the current CPT
	int getNumberOfCombinations(std::vector<VarStates>& states) {
		int numCombinations = 1;
		for (auto it = states.begin(); it != states.end(); it++)
			numCombinations *= it->m_nStates;

		return numCombinations;
	}

	//splits the string of parents into a vector containing their id and their number of states
	int splitParents(std::string& parents, std::vector<VarStates>& variablesOrder, std::shared_ptr<BayesianNetwork<T>> bn) {
		if (parents.size() == 0) return 0;
		int pos;
		while ((pos = parents.find(" ")) != std::string::npos) {
			std::string parent = parents.substr(0, pos);
			parents = parents.substr(pos + 1);

			NodeId id = bn->idFromName(parent);

			int nStates = bn->getVariableStatesNum(id);

			variablesOrder.emplace_back(id, nStates);
		}

		NodeId id = bn->idFromName(parents);

		int nStates = bn->getVariableStatesNum(id);

		variablesOrder.emplace_back(id, nStates);

		return variablesOrder.size();
	}

	//splits the string containing the probability distribution and adds each combination of states with its probability to the CPT
	void splitProbabilities(std::string& probDistribution, std::vector<VarStates>& variablesOrder, CPT<T>& cpt) {
		int pos;

		std::vector<int> combination;

		getInitialCombination(variablesOrder, combination);

		while ((pos = probDistribution.find(" ")) != std::string::npos) {
			double prob = std::stod(probDistribution.substr(0, pos));
			probDistribution = probDistribution.substr(pos + 1);

			cpt.addProbability(combination, prob);

			getNextCombination(variablesOrder, combination);
		}

		double prob = std::stod(probDistribution);

		cpt.addProbability(combination, prob);
	}

	//splits the string containing the resulting states into a vector of string. returns the number of states found.
	//used for deterministic variables
	int splitResultingStates(std::string& resultingStates, std::vector<std::string>& resultingStatesSplitted) {
		if (resultingStates.size() == 0) return 0;
		int pos;
		while ((pos = resultingStates.find(" ")) != std::string::npos) {
			std::string state = resultingStates.substr(0, pos);
			resultingStates = resultingStates.substr(pos + 1);

			resultingStatesSplitted.push_back(state);
		}

		resultingStatesSplitted.push_back(resultingStates);

		return resultingStatesSplitted.size();
	}
};