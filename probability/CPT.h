#pragma once

#include <map>
#include <set>
#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include <atomic>
#include <functional>
#include <algorithm>

#include "../graphs/graphStructure/GraphElements.h"
#include "../tools/COW.hpp"

#define SPARSETHRESHOLD 0.7

//struct used to store information about the value of the variable and its probability
template <typename T = float>
struct StateProb {
	int m_state;
	T m_prob;

	StateProb(int state, T prob) : m_state(state), m_prob(prob) {}

	bool operator==(const StateProb<T>& sp) const {
		return m_state == sp.m_state && m_prob == sp.m_prob;
	}

	StateProb<T> operator+(const StateProb<T>& sp) const {
		return StateProb<T>(m_state, m_prob + sp.m_prob);
	}

	StateProb<T>& operator+=(const StateProb<T>& sp) {
		m_prob += sp.m_prob;
		return *this;
	}

	StateProb<T>& operator/=(const int div) {
		m_prob /= div;
		return *this;
	}
};

//struct used to store information about the id of a variable its number of states
struct VarStates {
	NodeId m_id;
	int m_nStates;

	VarStates() : m_id(-1), m_nStates(0) {}

	VarStates(NodeId id, int nStates) : m_id(id), m_nStates(nStates) {}

	bool operator<(const VarStates& vs) const {
		return m_id < vs.m_id;
	}
};

//struct used to store information about the evidence introduced in the network.
//m_id: id of the variable
//m_state: state to which the variable is istantiated
struct Evidence {
	NodeId m_id;
	int m_state;

	Evidence(NodeId id, int state) : m_id(id), m_state(state) {}

	bool operator<(const Evidence& e) const {
		return m_id < e.m_id;
	}
};

//struct used to store the probability distribution of the CPT
template <typename T = float>
struct CPTData {
	std::map<std::vector<int>, T> m_cpt;
};

template <class T = float>
class CPT : private COW<CPTData<T>> {

public:
	//default CPT constructor without parameters
	CPT() : m_var(-1), m_sparseProb(-1), m_sparseCount(-1), m_instantiated(false) {
		this->construct();
	}

	//CPT constructor used during reading of source file containing the bayesian network
	CPT(const std::vector<std::string>& stateNames) : m_var(-1), m_sparseProb(-1), m_sparseCount(-1), m_instantiated(false) {
		this->construct();

		for (int i = 0; i < stateNames.size(); i++) {
			m_valueStateName.emplace(i, stateNames[i]);
		}
	}

	//CPT constructor used to initialize the CPT's info but not the probability distribution
	CPT(const std::map<int, std::string>& valueStates, T sparseProb, int sparseCount, int CPTSize) :
		m_var(-1), m_sparseProb(sparseProb), m_sparseCount(sparseCount), m_valueStateName(valueStates), m_instantiated(false) {
        this->construct();
	}

	//CPT constructor used to compute the potential of multiple CPTs.
	//It receives the evidences introduced in the bayesian network in order to calculate the combinations of states for each variable
	CPT(std::vector<CPT<T>>& CPTs, std::vector<Evidence>& evidences) : m_var(-1), m_sparseProb(-1), m_sparseCount(-1), m_instantiated(false) {
        this->construct();

		std::map<NodeId, int> evidenceMap;

		for (int i = 0; i < evidences.size(); i++)
			evidenceMap.emplace(evidences[i].m_id, evidences[i].m_state);

		getNodeStates(CPTs);

		std::vector<int> combination;

		getInitialCombination(evidenceMap, combination);

		int numOfCombinations = getNumberOfCombinations(evidenceMap);

		auto itHint = this->ptr()->m_cpt.begin();

		for (int i = 0; i < numOfCombinations; i++) {
			T prob = 1;

			for (auto it2 = CPTs.begin(); it2 != CPTs.end(); it2++) {

				//for each combination, computes the product of the corresponding probabilities from each parent CPT
				std::vector<int> partialCombination;
				partialCombination.reserve(m_variablesOrder.size());

				//creates the partial combination needed to get the probability from the parent CPT it2
				for (auto it3 = it2->m_variablesOrder.begin(); it3 != it2->m_variablesOrder.end() && prob > 0; it3++) {

					int pos = getVariablePosition(it3->m_id);

					partialCombination.push_back(combination[pos]);
				}

				prob *= it2->getProbability(partialCombination);

				partialCombination.clear();
			}

            this->ptr()->m_cpt.emplace_hint(itHint, combination, prob);

			itHint = this->ptr()->m_cpt.end();

			getNextCombination(evidenceMap, combination);
		}

		checkSparseCPT();
	}

	~CPT() {}

	bool operator==(const CPT<T>& c) const {
		if (this->ptr()->m_cpt.size() > 0)
			return this->ptr()->m_cpt == c.ptr()->m_cpt;
		else
			return floatingPointCompare(m_sparseProb, c.m_sparseProb);
	}

	////divides each probability in the CPT by the sparseProbability of another CPT which has been marginalized
	//CPT<T>& operator/=(const CPT<T>& c) {
	//	for (auto it = ptr()->m_cpt.begin(); it != ptr()->m_cpt.end(); it++) {
	//		it->second /= c.m_sparseProb;
	//	}

	//	if (m_sparseProb > 0)
	//		m_sparseProb /= c.m_sparseProb;

	//	return *this;
	//}

	//uses the same data contained in c for the CPTData of this CPT
	void duplicateCPT(CPT& c) {
		duplicate(c);
	}

	//adds variables order to the CPT
	void addVariablesOrder(const std::vector<VarStates>&& variablesOrder) {
		m_variablesOrder = std::move(variablesOrder);
	}

	////adds probability distribution for a combination of the conditional variables to the CPT
	//void addProbabilities(const std::vector<int>& variablesCombination, const T prob) {
	//	clone_if_needed();
	//	ptr()->m_cpt.emplace(std::make_pair(variablesCombination, prob));
	//}

	//adds probability distribution for a combination of variables to the CPT
	void addProbability(const std::vector<int>& variablesCombination, const T prob) {
        this->clone_if_needed();

        this->ptr()->m_cpt.emplace(variablesCombination, prob);
	}

	//adds resulting states to the CPT
	void addResultingStates(const std::vector<std::string>&& resultingStates) {
		m_resultingStates = std::move(resultingStates);
	}

	//checks if the base CPT is sparse
	void checkSparseCPT() {
		m_CPTSize = this->ptr()->m_cpt.size();

		if (m_CPTSize == 1) {
			m_sparseProb = this->ptr()->m_cpt.begin()->second;
            this->ptr()->m_cpt.clear();
			return;
		}

		std::map<T, int> valueCount;

		for (auto it = this->ptr()->m_cpt.begin(); it != this->ptr()->m_cpt.end(); it++) {
			auto p = valueCount.find(it->second);
			if (p != valueCount.end()) {
				p->second++;

				//converts the matrix into a sparse representation if it is sparse
				if (p->second > (m_CPTSize * SPARSETHRESHOLD)) {
					m_sparseProb = p->first;
					m_sparseCount = p->second;
					convertCPTToSparseCPT();
					return;
				}
			}
			else
				valueCount.emplace(it->second, 1);
		}
	}

	//void displayCPT() const {
	//	for (auto it = ptr()->m_cpt.begin(); it != ptr()->m_cpt.end(); it++) {
	//		std::cout << "combination";
	//		for (auto it2 = it->first.begin(); it2 != it->first.end(); it2++) {
	//			std::string s = std::to_string(*it2);
	//			std::cout << s << ",";
	//		}

	//		for (auto it2 = it->second.begin(); it2 != it->second.end(); it2++) {
	//			std::cout << it2->m_prob << ",";
	//		}

	//		std::cout << std::endl;
	//	}
	//}

	int getCPTId() {
		return m_var;
	}

	//void getEmptyProbabilityVector(std::vector<StateProb<T>>& probVec) {
	//	for (auto it = m_valueStateName.begin(); it != m_valueStateName.end(); it++) {
	//		probVec.emplace_back(it->first, 0);
	//	}
	//}

	//returns a reference to the list of variables of the CPT
	std::vector<VarStates>* getVariables() {
		return &m_variablesOrder;
	}

	//finds the position of a variable inside the vector of variables
	int getVariablePosition(const NodeId var) {
		for (int i = 0; i < m_variablesOrder.size(); i++) {
			if (m_variablesOrder[i].m_id == var) return i;
		}
		return -1;
	}

	//returns the probability of a combination of variables
	T getProbability(std::vector<int>& combination) const {
		auto it = this->ptr()->m_cpt.find(combination);

		//if the combination is not present in the CPT returns the sparse probability
		if (it == this->ptr()->m_cpt.end())
			return m_sparseProb;
		else {
			return it->second;
		}
	}

	//returns the probability distribution of a combination of parent variables
	std::vector<StateProb<T>> getProbabilityDistribution(std::vector<int>& variablesCombination) const {
		std::vector<int> combination(variablesCombination);
		combination.push_back(0);

		std::vector<StateProb<T>> probDistribution;

		while (combination[m_variablesOrder.size() - 1] < m_variablesOrder[m_variablesOrder.size() - 1].m_nStates) {
			auto it = this->ptr()->m_cpt.find(combination);

			if (it != this->ptr()->m_cpt.end())
				probDistribution.emplace_back(combination[m_variablesOrder.size() - 1], it->second);
			else
				probDistribution.emplace_back(combination[m_variablesOrder.size() - 1], m_sparseProb);

			combination[m_variablesOrder.size() - 1]++;
		}

		return probDistribution;
	}

	T getSparseProb() const {
		return m_sparseProb;
	}

	int getStatesNum() const {
		return m_valueStateName.size();
	}

	//instantiate the variable to the evidence introduced.
	//returns a new CPT instantiated to evidence e
	CPT<T> instantiateVariable(const Evidence e) {
		int pos = getVariablePosition(e.m_id);

		if (pos != -1) {

			std::map<int, std::string> instantiatedMap(m_valueStateName);

			//if the variable to be instantiated is the conditioned variable, then all the states are removed except for the one instantiated
			if (e.m_id == m_var) {
				for (auto it = instantiatedMap.begin(); it != instantiatedMap.end();) {
					if (it->first != e.m_state)
						it = instantiatedMap.erase(it);
					else
						it++;
				}
			}

			CPT<T> instantiatedCPT(instantiatedMap, m_sparseProb, m_sparseCount, m_CPTSize / m_valueStateName.size());

			instantiatedCPT.m_variablesOrder = m_variablesOrder;

			for (auto it = this->ptr()->m_cpt.begin(); it != this->ptr()->m_cpt.end(); it++) {
				if (it->first[pos] == e.m_state) {
					instantiatedCPT.ptr()->m_cpt.emplace(it->first, it->second);
				}
			}

			instantiatedCPT.m_resultingStates = m_resultingStates;
			instantiatedCPT.m_instantiated = true;

			//checks if the new CPT is derived from a sparse CPT, then updates the representation
			if (m_sparseProb > -1)
				instantiatedCPT.convertCPTToSparseCPT();
			//if the original was not sparse, checks is the new one is sparse
			else
				instantiatedCPT.checkSparseCPT();

			return instantiatedCPT;
		}

		return *this;
	}

	bool isEmpty() {
		return this->ptr()->m_cpt.size() == 0 && floatingPointCompare(m_sparseProb, -1);
	}

	bool isInstantiated() {
		return m_instantiated;
	}

	bool isUnitary() {
		return floatingPointCompare(m_sparseProb, 1) && this->ptr()->m_cpt.size() == 0;
	}

	//returns a new CPT where the variable var has been marginalized out
	CPT<T> marginalizeVariable(const NodeId var) {
		int pos = getVariablePosition(var);

		if (pos != -1) {

			VarStates parent = m_variablesOrder[pos];

			CPT<T> marginalizedCPT(m_valueStateName, m_sparseProb, m_sparseCount, m_CPTSize);

			std::vector<VarStates> newParentsOrder;

			//copies the CPT's parents execpt var into a new vector
			std::copy(m_variablesOrder.begin(), m_variablesOrder.begin() + pos, std::back_inserter(newParentsOrder));
			std::copy(m_variablesOrder.begin() + pos + 1, m_variablesOrder.end(), std::back_inserter(newParentsOrder));

			marginalizedCPT.m_variablesOrder = std::move(newParentsOrder);

			for (auto it = this->ptr()->m_cpt.begin(); it != this->ptr()->m_cpt.end(); it++) {
				T prob = 0;

				std::vector<int> combination = it->first;

				std::vector<int> newCombination;

				//gets the initial parents' combination from the actual CPT and copies it into a new vector execpt parent var, which is located at position pos
				//inside the combination vector
				std::copy(combination.begin(), combination.begin() + pos, std::back_inserter(newCombination));
				std::copy(combination.begin() + pos + 1, combination.end(), std::back_inserter(newCombination));

				//if the combination thus found has been already added to the new CPT, it is skipped
				if (marginalizedCPT.ptr()->m_cpt.find(newCombination) == marginalizedCPT.ptr()->m_cpt.end()) {

					//sums all probability distribution for each value of var in the current combination
					while (combination[pos] < parent.m_nStates) {
						auto it2 = this->ptr()->m_cpt.find(combination);
						if (it2 != this->ptr()->m_cpt.end()) {
							prob += it2->second;
						}
						else {
							prob += m_sparseProb;
						}

						combination[pos]++;
					}

					marginalizedCPT.addProbability(newCombination, prob);
				}
			}

			marginalizedCPT.m_resultingStates = m_resultingStates;

			if (pos == m_variablesOrder.size() - 1)
				marginalizedCPT.checkSparseCPT();
			else if (m_sparseProb > -1)
				marginalizedCPT.convertCPTToSparseCPT();
			else
				marginalizedCPT.checkSparseCPT();

			return marginalizedCPT;
		}
		else {
			this;
		}

	}

	void setCPTId(const NodeId id) {
		m_var = id;
	}

	//updates probability distribution of a combination of variables
	void updateProbabilities(const std::vector<int>& combination, const T prob) {
        this->clone_if_needed();
		auto it = this->ptr()->m_cpt.find(combination);
		if (it != this->ptr()->m_cpt.end()) {
			it->second = prob;
		}
		else {
            this->ptr()->m_cpt.emplace(combination, prob);
			m_sparseCount--;
		}
	}

private:
	//converts the CPT into a sparse representation
	void convertCPTToSparseCPT() {

		for (auto it = this->ptr()->m_cpt.begin(); it != this->ptr()->m_cpt.end();) {
			if (floatingPointCompare(it->second, m_sparseProb))
				it = this->ptr()->m_cpt.erase(it);
			else
				it++;
		}

		if (this->ptr()->m_cpt.size() == 0)
            this->ptr()->m_cpt.clear();
	}

	bool floatingPointCompare(const T n1, const T n2) const {
		if (abs(n1 - n2) < 1.0e-5) return true;
		return abs(n1 - n2) < 1.05e-5 * std::max(abs(n1), abs(n2));
	}

	//given the evidences introduced into the network, inserts the initial combination into combination
	void getInitialCombination(std::map<NodeId, int>& evidenceMap, std::vector<int>& combination) {
		for (int i = 0; i < m_variablesOrder.size(); i++) {
			if (evidenceMap.find(m_variablesOrder[i].m_id) == evidenceMap.end()) {
				combination.push_back(0);
			}
			else {
				combination.push_back(evidenceMap.find(m_variablesOrder[i].m_id)->second);
			}
		}
	}

	//given the evidences introduced into the network, inserts in combination the next one to the that already stored
	void getNextCombination(std::map<NodeId, int>& evidenceMap, std::vector<int>& combination) {
		int i = combination.size() - 1;

		if (evidenceMap.size() > 0) {
			while (true) {
				if (evidenceMap.find(m_variablesOrder[i].m_id) == evidenceMap.end()) break;
				else i--;
			}
		}

		int inc = 1;

		for (i; i > -1; i--) {
			if (evidenceMap.size() > 0 && evidenceMap.find(m_variablesOrder[i].m_id) == evidenceMap.end()) {
				combination[i] += inc;
				combination[i] = combination[i] % m_variablesOrder[i].m_nStates;
				if (combination[i] != 0) break;
			}
			else {
				combination[i] += inc;
				combination[i] = combination[i] % m_variablesOrder[i].m_nStates;
				if (combination[i] != 0) break;
			}
		}
	}

	//given the evidences introduced into the network, returns the number of combinations of states for the current CPT
	int getNumberOfCombinations(std::map<NodeId, int>& evidenceMap) {
		int numCombinations = 1;
		for (auto it = m_variablesOrder.begin(); it != m_variablesOrder.end(); it++) {
			if (evidenceMap.find(it->m_id) == evidenceMap.end())
				numCombinations *= it->m_nStates;
		}

		return numCombinations;
	}

	//generates the variables order for this CPT taking the parent CPTs as source
	void getNodeStates(std::vector<CPT<T>>& CPTs) {
		std::set<VarStates> temp;

		for (auto it = CPTs.begin(); it != CPTs.end(); it++) {
			for (auto it2 = it->m_variablesOrder.begin(); it2 != it->m_variablesOrder.end(); it2++) {
				auto ins = temp.insert(*it2);

				if (ins.second)
					m_variablesOrder.push_back(*it2);
			}
		}
	}

	NodeId m_var;
	std::vector<VarStates> m_variablesOrder;
	std::vector<std::string> m_resultingStates;
	std::map<int, std::string> m_valueStateName;
	T m_sparseProb;
	int m_CPTSize;
	int m_sparseCount;
	bool m_instantiated;
};