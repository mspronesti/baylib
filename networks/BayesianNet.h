#pragma once

#include <iostream>

#include "../graphs/DAG.h"
#include "../probability/CPT.h"
#include "../tools/VariableNodeMap.h"

template <class T = float>
class BayesianNetwork {
public:

	//constructor
	BayesianNetwork() : m_bn(std::make_shared<DAG>()) {}

	//copy constructor
	BayesianNetwork(const BayesianNetwork& bn) {
		m_bn = bn.m_bn;
	}

	//destructor
	~BayesianNetwork() {
		m_bn.reset();
	}

	//copy operator
	BayesianNetwork& operator=(const BayesianNetwork& bn) {
		if (this != &bn) {
			m_bn = bn.m_bn;
		}

		return *this;
	}

	//boolean operator
	bool operator==(const BayesianNetwork& bn) const {
		return  *m_bn == *(bn.m_bn);
	}

	//adds and arc to the old_graph
	void addArc(NodeId node1, NodeId node2) {
		m_bn->addArc(node1, node2);
	}

	//adds arcs to the graphs using the dependencies obtained from the CPTS
	void addArcsFromCPTs() {
		for (auto it = m_cpt.begin(); it != m_cpt.end(); it++) {
			std::vector<VarStates>* parents = it->second.getVariables();

			for (auto it2 = parents->begin(); it2 != std::prev(parents->end()); it2++) {
				m_bn->addArc(it2->m_id, it->first);
			}
		}

		m_bn->setNumberOfNodes(m_vnm.getNumberOfVariables());
	}

	//void addProbabilities(const NodeId n, const std::vector<int>& var, const std::vector<StateProb<T>>& probs) {
	//	m_cpt.find(n)->second.addProbability(var, probs);
	//}

	//adds an empty variable to the old_graph, wihtout connecting it.
	//used for debug purposes
	void addVariable(const std::string& variableName) {
		NodeId id = m_vnm.addVariable(variableName);
		CPT<T> cpt;
		m_cpt.emplace(id, cpt);
	}

	//adds a variable to the old_graph, without connecting it
	//used for debug purposes
	void addVariable(const std::string& variableName, CPT<T> cpt) {
		NodeId id = m_vnm.addVariable(variableName);

		cpt.setCPTId(id);

		for (auto it = m_cpt.begin(); it != m_cpt.end(); it++) {
			if (it->second == cpt) {
				cpt.duplicateCPT(it->second);
				m_nodeWithSameCPT.emplace(id, it->first);
				break;
			}
		}

		m_cpt.emplace(id, cpt);
	}

	//adds a variable to the old_graph, using the same CPT of another variable, without connecting it
	//used for debug purposes
	void addVariable(NodeId source, const std::string& variableName) {
		m_vnm.addVariable(variableName);
		NodeId id = m_vnm.idFromName(variableName);
		CPT<T> sourceCPT = m_cpt.find(source)->second;
		m_cpt.emplace(id, sourceCPT);

		if (m_nodeWithSameCPT.find(source) != m_nodeWithSameCPT.end()) {
			m_nodeWithSameCPT.emplace(id, m_nodeWithSameCPT.find(source)->second);
		}
		else {
			m_nodeWithSameCPT.emplace(id, source);
		}
	}

    void checkSparseCPTs() {
	    checkSparseCPTsUnparallel();
    }

	void checkSparseCPTsUnparallel() {
		for (auto it = m_cpt.begin(); it != m_cpt.end(); it++) {
			if (m_nodeWithSameCPT.find(it->first) == m_nodeWithSameCPT.end()) {
				it->second.checkSparseCPT();
			}
		}
	}


	void checkSparseCPT(const NodeId n) {
		m_cpt.find(n)->second.checkSparseCPT();
	}

	//onyl for debug purposes at the moment
	//void displayCPT(const NodeId n) const {
	//	m_cpt.find(n)->second.displayCPT();
	//}

	//returns the id of a variable given its name
	NodeId idFromName(const std::string& variableName) {
		return m_vnm.idFromName(variableName);
	}


	int getNextVariableId() {
		return m_vnm.getNumberOfVariables();
	}

	//wrapper function that returns the number of states of a given variable node
	int getVariableStatesNum(const NodeId node) {
		return m_cpt.at(node).getStatesNum();
	}

	bool hasChildren(const NodeId node) {
		return m_bn->hasChildren(node);
	}

	//wrapper function that removes a node from the DAG
	void removeArc(NodeId node1, NodeId node2) {
		m_bn->removeArc(node1, node2);
	}

	//wrapper function for changing the probability of a combination of states inside the CPT of node n
	void updateProbabilities(const NodeId n, const std::vector<int>& var, T prob) {
		m_cpt.find(n)->second.updateProbabilities(var, prob);
	}

	//returns the CPT of node n
	CPT<T> getCPT(const NodeId n) {
		return m_cpt.at(n);
	}

private:
	std::shared_ptr<DAG> m_bn;
	std::map<NodeId, CPT<T>> m_cpt;
	std::map<NodeId, NodeId> m_nodeWithSameCPT;
	VariableNodeMap m_vnm;
};