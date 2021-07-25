//
// Created by elle on 24/07/21.
//

#ifndef BAYESIAN_INFERRER_CPT_H
#define BAYESIAN_INFERRER_CPT_H

#include <utility>
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <string>
#include <atomic>
#include <functional>
#include <algorithm>

#include "../graphs/graphStructure/GraphElements.h"
#include "../tools/COW.h"

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
    std::vector<T> m_cpt;
    std::set<int> m_combinationsRemained;
    T m_sparseProb;
    int m_CPTSize;
    int m_sparseCount;
};



template <class T = float>
class CPT : private COW<CPTData<T>> {
public:
    using COW<CPTData<T>>::construct;
    using COW<CPTData<T>>::ptr;
    using COW<CPTData<T>>::clone_if_needed;

    //default CPT constructor without parameters
    CPT() : m_var(-1), m_instantiated(false) {
        construct();

        ptr()->m_sparseProb = -1;
        ptr()->m_sparseCount = -1;
    }

    //CPT constructor used during reading of source file containing the bayesian network
    explicit CPT(const std::vector<std::string>& stateNames) : m_var(-1), m_instantiated(false) {
        construct();

        ptr()->m_sparseProb = -1;
        ptr()->m_sparseCount = -1;

        for (int i = 0; i < stateNames.size(); i++) {
            m_valueStateName.emplace(i, stateNames[i]);
        }
    }

    //CPT constructor used to initialize the CPT's info but not the probability distribution
    CPT(std::map<int, std::string>  valueStates, T sparseProb, int sparseCount, int CPTSize) :
            m_var(-1), m_valueStateName(std::move(valueStates)), m_instantiated(false) {
        construct();

        ptr()->m_sparseProb = sparseProb;
        ptr()->m_sparseCount = sparseCount;
        ptr()->m_CPTSize = CPTSize;
    }

    //CPT constructor used to compute the potential of multiple CPTs.
    //It receives the evidences introduced in the bayesian network in order to calculate the combinations of states for each variable
    CPT(std::vector<CPT<T>>& CPTs, std::vector<Evidence>& evidences) : m_var(-1), m_instantiated(false) {
        construct();

        ptr()->m_sparseProb = -1;
        ptr()->m_sparseCount = -1;

        for (auto & evidence : evidences)
            m_evidences.emplace(evidence.m_id, evidence.m_state);

        getNodeStates(CPTs);

        std::vector<int> combination;

        getInitialCombination(combination);

        int numOfCombinations = getNumberOfCombinations();

        ptr()->m_cpt.reserve(numOfCombinations);

        for (int i = 0; i < numOfCombinations; i++) {
            T prob = 1;

            for (auto it2 = CPTs.begin(); it2 != CPTs.end(); it2++) {

                //for each combination, computes the product of the corresponding probabilities from each parent CPT
                std::vector<int> partialCombination;
                partialCombination.reserve(m_variablesOrder.size());

                //creates the partial combination needed to get the probability from the parent CPT it2
                for (auto it3 = it2->m_variablesOrder.begin(); it3 != it2->m_variablesOrder.end(); it3++) {

                    int pos = getVariablePosition(it3->m_id);

                    partialCombination.push_back(combination[pos]);
                }

                prob *= it2->getProbability(partialCombination);

                if (floatingPointCompare(prob, 0)) break;

                partialCombination.clear();
            }

            ptr()->m_cpt.push_back(prob);

            getNextCombination(combination);
        }

        checkSparseCPT();
    }

    ~CPT() {}

    bool operator==(const CPT<T>& c) const {
        if (ptr()->m_cpt.size() > 0 && floatingPointCompare(ptr()->m_sparseProb, -1))
            return ptr()->m_cpt == c.ptr()->m_cpt;
        else if (ptr()->m_cpt.size() > 0 && ptr()->m_sparseProb > -1) {
            return ptr()->m_sparseProb == c.ptr()->m_sparseProb &&
                   ptr()->m_CPTSize == c.ptr()->m_CPTSize && ptr()->m_cpt == c.ptr()->m_cpt;
        }
        else
            return floatingPointCompare(ptr()->m_sparseProb, c.ptr()->m_sparseProb);
    }

    //uses the same data contained in c for the CPTData of this CPT
    void duplicateCPT(CPT& c) {
        this->duplicate(c);
    }

    //adds variables order to the CPT
    void addVariablesOrder(std::vector<VarStates>&& variablesOrder) {
        m_variablesOrder = std::move(variablesOrder);
    }

    void addProbabilities(std::vector<T>& probs) {
        clone_if_needed();

        ptr()->m_cpt = std::move(probs);
    }

    //adds probability distribution for a combination of variables to the CPT
    void addProbability(const T prob) {
        clone_if_needed();
        ptr()->m_cpt.push_back(prob);
    }

    //adds resulting states to the CPT
    void addResultingStates(const std::vector<std::string>&& resultingStates) {
        m_resultingStates = resultingStates;
    }

    //checks if the base CPT is sparse
    void checkSparseCPT() {
        std::map<T, int> valueCount;

        if (ptr()->m_cpt.size() > 0) {

            ptr()->m_CPTSize = ptr()->m_cpt.size();

            for (int i = 0; i < ptr()->m_cpt.size(); i++) {
                auto p = valueCount.find(ptr()->m_cpt[i]);
                if (p != valueCount.end()) {
                    p->second++;

                    //converts the matrix into a sparse representation if it is sparse
                    if (p->second > (ptr()->m_CPTSize * SPARSETHRESHOLD)) {
                        ptr()->m_sparseProb = p->first;
                        ptr()->m_sparseCount = p->second;
                        convertCPTToSparseCPT();
                        return;
                    }
                }
                else
                    valueCount.emplace(ptr()->m_cpt[i], 1);
            }
        }
    }

    //calculate the index at which the combination is stored in the CPT
    int findCombinationPosition(const std::vector<int>& combination) const {
        int pos = 0;

        int base = 1;

        for (int i = combination.size() - 1; i > -1; i--) {
            pos += (combination[i]) * base;

            base *= m_variablesOrder[i].m_nStates;
        }

        return pos;
    }

    //void displayCPT() const {
    //	for (auto it = ptr()->m_sparseCPT.begin(); it != ptr()->m_sparseCPT.end(); it++) {
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
        int pos = findCombinationPosition(combination);

        if (ptr()->m_sparseProb > -1) {
            auto it = ptr()->m_combinationsRemained.find(pos);

            if (it == ptr()->m_combinationsRemained.end()) return ptr()->m_sparseProb;
            else return ptr()->m_cpt[std::distance(ptr()->m_combinationsRemained.begin(), it)];
        }
        else
            return ptr()->m_cpt[pos];
    }

    //returns the probability distribution of a combination of variables
    std::vector<StateProb<T>> getProbabilityDistribution(std::vector<int>& variablesCombination) const {
        std::vector<int> combination(variablesCombination);
        combination.push_back(0);

        std::vector<StateProb<T>> probDistribution;

        while (combination[m_variablesOrder.size() - 1] < m_variablesOrder[m_variablesOrder.size() - 1].m_nStates) {
            int pos = findCombinationPosition(combination);

            if (ptr()->m_sparseProb > -1) {
                auto it = ptr()->m_combinationsRemained.find(pos);

                if (it == ptr()->m_combinationsRemained.end()) probDistribution.emplace_back(combination[m_variablesOrder.size() - 1], ptr()->m_sparseProb);
                else probDistribution.emplace_back(combination[m_variablesOrder.size() - 1], ptr()->m_cpt[std::distance(ptr()->m_combinationsRemained.begin(), it)]);
            }
            else
                probDistribution.emplace_back(combination[m_variablesOrder.size() - 1], ptr()->m_cpt[pos]);

            combination[m_variablesOrder.size() - 1]++;
        }

        return probDistribution;
    }

    T getSparseProb() const {
        return ptr()->m_sparseProb;
    }

    int getStatesNum() const {
        return m_valueStateName.size();
    }

    //instantiate the variable to the evidence introduced.
    //returns a new CPT instantiated to evidence e
    CPT<T> instantiateVariable(const Evidence e) {
        int variablePos = getVariablePosition(e.m_id);

        if (variablePos != -1) {

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

            CPT<T> instantiatedCPT(instantiatedMap, ptr()->m_sparseProb, ptr()->m_sparseCount, ptr()->m_CPTSize / m_valueStateName.size());

            instantiatedCPT.m_variablesOrder = m_variablesOrder;
            instantiatedCPT.m_variablesOrder[variablePos].m_nStates = 1;
            instantiatedCPT.m_evidences.emplace(e.m_id, e.m_state);

            for (auto it = m_evidences.begin(); it != m_evidences.end(); it++) {
                instantiatedCPT.m_evidences.insert(*it);
            }

            int nStatesOfFollowingVariables = 1;

            //finds the number of combinations of variables following the one to be marginalized inside the variables order
            for (int i = variablePos + 1; i < m_variablesOrder.size(); i++) {
                nStatesOfFollowingVariables *= m_variablesOrder[i].m_nStates;
            }

            int nStatesOfPrecedingVariables = 1;

            //finds the number of combinations of variables preceding the one to be marginalized inside the variables order
            for (int i = 0; i < variablePos; i++) {
                nStatesOfPrecedingVariables *= m_variablesOrder[i].m_nStates;
            }

            int nCombinationsToSkip = nStatesOfFollowingVariables * m_variablesOrder[variablePos].m_nStates;

            int initialPos = nStatesOfFollowingVariables * e.m_state;

            //stores the new position pf the remaining combinations in case the original CPT is sparse
            int newPos = 0;

            for (int i = 0; i < nStatesOfPrecedingVariables; i++) {
                int pos = initialPos;

                for (int j = 0; j < nStatesOfFollowingVariables; j++) {
                    if (ptr()->m_sparseProb > -1) {
                        auto it = ptr()->m_combinationsRemained.find(pos + j);

                        if (it != ptr()->m_combinationsRemained.end()){
                            instantiatedCPT.ptr()->m_cpt.push_back(ptr()->m_cpt[std::distance(ptr()->m_combinationsRemained.begin(), it)]);
                            instantiatedCPT.ptr()->m_combinationsRemained.insert(newPos);
                        }

                        newPos++;
                    }
                    else
                        instantiatedCPT.ptr()->m_cpt.push_back(ptr()->m_cpt[pos + j]);
                }

                initialPos += nCombinationsToSkip;
            }

            instantiatedCPT.m_resultingStates = m_resultingStates;
            instantiatedCPT.m_instantiated = true;

            //if the original was not sparse, checks is the new one is sparse
            if (ptr()->m_sparseProb == -1)
                instantiatedCPT.checkSparseCPT();


            return instantiatedCPT;
        }

        return *this;
    }

    bool isEmpty() {
        return ptr()->m_cpt.size() == 0 && floatingPointCompare(ptr()->m_sparseProb, -1);
    }

    bool isInstantiated() {
        return m_instantiated;
    }

    bool isUnitary() {
        return floatingPointCompare(ptr()->m_sparseProb, 1) && ptr()->m_cpt.size() == 0;
    }

    //returns a new CPT where the variable var has been marginalized out
    CPT<T> marginalizeVariable(const NodeId var) {
        int varPos = getVariablePosition(var);

        if (varPos != -1) {

            //if the marginalized variable is the conditioned variable, the resulting CPT is unitary if it is not instantied
            if (var == m_var && m_evidences.find(var) == m_evidences.end()) {
                return CPT<T>(m_valueStateName, 1, 0, 0);
            }
            else{

                CPT<T> marginalizedCPT(m_valueStateName, ptr()->m_sparseProb, ptr()->m_sparseCount, ptr()->m_CPTSize);

                std::vector<VarStates> newParentsOrder;

                //copies the CPT's parents execpt var into a new vector
                std::copy(m_variablesOrder.begin(), m_variablesOrder.begin() + varPos, std::back_inserter(newParentsOrder));
                std::copy(m_variablesOrder.begin() + varPos + 1, m_variablesOrder.end(), std::back_inserter(newParentsOrder));

                marginalizedCPT.m_variablesOrder = std::move(newParentsOrder);
                marginalizedCPT.m_evidences = m_evidences;

                VarStates variable = m_variablesOrder[varPos];

                int nStatesOfFollowingVariables = 1;

                //finds the number of combinations of variables following the one to be marginalized inside the variables order
                for (int i = varPos + 1; i < m_variablesOrder.size(); i++) {
                    nStatesOfFollowingVariables *= m_variablesOrder[i].m_nStates;
                }

                int nStatesOfPrecedingVariables = 1;

                //finds the number of combinations of variables preceding the one to be marginalized inside the variables order
                for (int i = 0; i < varPos; i++) {
                    nStatesOfPrecedingVariables *= m_variablesOrder[i].m_nStates;
                }

                int initialPos = 0;

                //moves initialPos to the next combination of preceding variables when all
                //combinations of following variables have been explored
                for (int l = 0; l < nStatesOfPrecedingVariables; l++) {

                    initialPos = nStatesOfFollowingVariables * m_variablesOrder[varPos].m_nStates * l;

                    //explores all combinations of following variables
                    for (int k = 0; k < nStatesOfFollowingVariables; k++) {

                        int pos = initialPos;

                        pos += k;

                        T prob = 0;

                        for (int j = 0; j < variable.m_nStates; j++) {
                            if (ptr()->m_sparseProb > -1) {
                                auto it = ptr()->m_combinationsRemained.find(pos);

                                if (it == ptr()->m_combinationsRemained.end()) prob += ptr()->m_sparseProb;
                                else prob += ptr()->m_cpt[std::distance(ptr()->m_combinationsRemained.begin(), it)];
                            }
                            else
                                prob += ptr()->m_cpt[pos];

                            pos += nStatesOfFollowingVariables;
                        }

                        marginalizedCPT.ptr()->m_cpt.push_back(prob);
                    }

                }

                marginalizedCPT.m_resultingStates = m_resultingStates;

                if (ptr()->m_sparseProb > -1)
                    marginalizedCPT.convertCPTToSparseCPT();
                else
                    marginalizedCPT.checkSparseCPT();

                return marginalizedCPT;
            }
        }
        else {
            this;
        }

    }

void marginalizeVariableNonSparseCPT(CPT<T>& marginalizedCPT, const int varPos) {
        VarStates variable = m_variablesOrder[varPos];

        int nStates = 1;

        for (int i = varPos + 2; i < m_variablesOrder.size(); i++) {
            nStates *= m_variablesOrder[i].m_nStates;
        }

        int nPrecStates = 1;

        for (int i = 0; i < varPos; i++) {
            nPrecStates *= m_variablesOrder[i].m_nStates;
        }

        int initialPos = 0;

        for (int l = 0; l < nPrecStates; l++) {

            T prob = 0;

            initialPos += (nStates * m_variablesOrder[varPos].m_nStates * l);

            for (int k = 0; k < nStates; k++) {

                int pos = initialPos;

                pos += k;

                for (int j = 0; j < variable.m_nStates; j++) {
                    prob += ptr()->m_cpt[pos];

                    pos += nStates;
                }

                marginalizedCPT.ptr()->m_cpt.push_back(prob);
            }

        }
    }


    //void marginalizeVariableSparseCPT(CPT<T>& marginalizedCPT, const int pos) {
    //	VarStates variable = m_variablesOrder[pos];

    //	for (auto it = ptr()->m_sparseCPT.begin(); it != ptr()->m_sparseCPT.end(); it++) {
    //		T prob = 0;

    //		std::vector<int> combination = it->first;

    //		std::vector<int> newCombination;

    //		//gets the initial parents' combination from the actual CPT and copies it into a new vector execpt parent var, which is located at position pos
    //		//inside the combination vector
    //		std::copy(combination.begin(), combination.begin() + pos, std::back_inserter(newCombination));
    //		std::copy(combination.begin() + pos + 1, combination.end(), std::back_inserter(newCombination));

    //		//if the combination thus found has been already added to the new CPT, it is skipped
    //		if (marginalizedCPT.ptr()->m_sparseCPT.find(newCombination) == marginalizedCPT.ptr()->m_sparseCPT.end()) {

    //			//sums all probability distribution for each value of var in the current combination
    //			while (combination[pos] < variable.m_nStates) {
    //				auto it2 = ptr()->m_sparseCPT.find(combination);
    //				if (it2 != ptr()->m_sparseCPT.end()) {
    //					prob += it2->second;
    //				}
    //				else {
    //					prob += ptr()->m_sparseProb;
    //				}

    //				combination[pos]++;
    //			}

    //			marginalizedCPT.addProbability(newCombination, prob);
    //		}
    //	}
    //}

    void setCPTId(const NodeId id) {
        m_var = id;
    }

    //updates probability distribution of a combination of variables
    void updateProbabilities(const std::vector<int>& combination, const T prob) {
        clone_if_needed();

        int pos = findCombinationPosition(combination);

        if (ptr()->m_sparseProb > -1) {
            auto res = ptr()->m_combinationsRemained.insert(pos);

            if (res.second) {
                ptr()->m_cpt.insert(ptr()->m_cpt.begin() + std::distance(ptr()->m_combinationsRemained.begin(), res.first), prob);
            }
        }
        else
            ptr()->m_cpt[pos] = prob;
    }

private:
    //converts the CPT into a sparse representation
    void convertCPTToSparseCPT() {
        std::vector<int> combination;

        getInitialCombination(combination);

        auto itHint = ptr()->m_combinationsRemained.begin();

        int i = 0;
        for (auto it = ptr()->m_cpt.begin(); it != ptr()->m_cpt.end(); i++) {
            if (floatingPointCompare(*it, ptr()->m_sparseProb))
                it = ptr()->m_cpt.erase(it);
            else {
                ptr()->m_combinationsRemained.insert(itHint, i);
                itHint = ptr()->m_combinationsRemained.end();
                it++;
            }

            getNextCombination(combination);
        }
    }

    bool floatingPointCompare(const T n1, const T n2) const {
        if (abs(n1 - n2) < 1.0e-5) return true;
        return abs(n1 - n2) < 1.05e-5 * std::max(abs(n1), abs(n2));
    }

    //given the evidences introduced into the network, inserts the initial combination into combination
    void getInitialCombination(std::vector<int>& combination) {
        for (int i = 0; i < m_variablesOrder.size(); i++) {
            if (m_evidences.find(m_variablesOrder[i].m_id) == m_evidences.end()) {
                combination.push_back(0);
            }
            else {
                combination.push_back(m_evidences.find(m_variablesOrder[i].m_id)->second);
            }
        }
    }

    //given the evidences introduced into the network, inserts in combination the next one to the that already stored
    void getNextCombination(std::vector<int>& combination) {
        int i = combination.size() - 1;

        if (m_evidences.size() > 0) {
            while (true) {
                if (m_evidences.find(m_variablesOrder[i].m_id) == m_evidences.end()) break;
                else i--;
            }
        }

        int inc = 1;

        for (i; i > -1; i--) {
            if (m_evidences.size() > 0 && m_evidences.find(m_variablesOrder[i].m_id) == m_evidences.end()) {
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
    int getNumberOfCombinations() {
        int numCombinations = 1;
        for (auto it = m_variablesOrder.begin(); it != m_variablesOrder.end(); it++) {
            if (m_evidences.find(it->m_id) == m_evidences.end())
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
    std::map<NodeId, int> m_evidences;
    bool m_instantiated;
};

#endif //BAYESIAN_INFERRER_CPT_H
