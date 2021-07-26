#include "DAG.h"

#include <algorithm>

DAG::DAG() : ArcSet(), m_reverseDAG(nullptr) {
}

DAG::DAG(const DAG& d) : ArcSet(d), m_reverseDAG(nullptr) {
}

DAG::~DAG() {
}

void DAG::addNodesToCheck(NodeId n, std::vector<NodeId>& nodeSetToCheck) {
	auto sourceNode = m_parents.find(n);

	if (sourceNode != m_parents.end()) {
		for (auto it = sourceNode->second.begin(); it != sourceNode->second.end(); it++) {
			nodeSetToCheck.push_back(*it);
		}
	}
}

std::shared_ptr<DAG> DAG::createReverseDAG() {
	std::call_once(m_reverseDAGCreated, [&]() {
		m_reverseDAG = std::make_shared<DAG>(*this);

		for (auto it = m_arcs.begin(); it != m_arcs.end(); it++) {
			m_reverseDAG->addArc(it->getTail(), it->getHead());
		}
		});

	return std::make_shared<DAG>(*m_reverseDAG);
}

void DAG::dConnectedNodes(NodeSet& J, std::shared_ptr<ArcPairSet> legalArcPairs, NodeSet& R) {
	//auto R = std::make_shared<NodeSet>();

	std::set<Arc> arcs(m_arcs);
	std::set<Arc> arcsForThisIteration;
	std::set<Arc> arcsFoundForNextIteration;

	NodeId S = getNumberOfNodes();

	//std::map<int, std::set<Arc>> arcLabel;
	//arcLabel.emplace(-1, m_arcs);

	//arcLabel.emplace(0, std::set<Arc>());
	for (auto x = J.begin(); x != J.end(); x++) {
		Arc a(S, *x);
		addArc(a);

		//arcLabel.at(0).insert(a);

		arcsForThisIteration.insert(a);

		R.insert(*x);
	}

	int i = 0;

	bool legalPairFound = true;

	while (legalPairFound) {
		//arcLabel.emplace(i + 1, std::set<Arc>());
		legalPairFound = false;
		//auto it_minus1 = arcLabel.at(-1);
		for (auto it = m_arcs.begin(); it != m_arcs.end();) {

			//auto it_ati = arcLabel.at(i);
			auto it2 = arcsForThisIteration.begin();

			for (it2; it2 != arcsForThisIteration.end(); it2++) {
				NodeId adj = it->isAdjacent(*it2);
				if (!it->isReverse(*it2) && adj != -1) {
					if (i == 0) {
						//arcLabel.at(i + 1).insert(*it);
						arcsFoundForNextIteration.insert(*it);
						legalPairFound = true;
						R.insert(it->getOther(adj));
						it = m_arcs.erase(it);
						break;
					}
					else {
						std::pair<Arc, Arc> aps(*it, *it2);
						std::pair<Arc, Arc> apsRev(*it2, *it);
						auto found = legalArcPairs->find(aps);
						auto foundRev = legalArcPairs->find(apsRev);
						if (found != legalArcPairs->end() || foundRev != legalArcPairs->end()) {
							legalPairFound = true;
							//arcLabel.at(i + 1).insert(*it);
							arcsFoundForNextIteration.insert(*it);
							R.insert(it->getOther(adj));
							it = m_arcs.erase(it);
							break;
						}
					}
				}
			}

			if (!legalPairFound || it2 == arcsForThisIteration.end()) it++;
		}

		arcsForThisIteration = std::move(arcsFoundForNextIteration);

		i++;
	}

	//return R;
}

std::shared_ptr<ArcPairSet> DAG::findHeadToHeadArcs(NodeId n) {
	auto headToHeadArcs = std::make_shared<ArcPairSet>();

	auto nParents = m_parents.find(n);

	if (nParents != m_parents.end() && nParents->second.size() > 1) {
		for (auto it = nParents->second.begin(); it != std::prev(nParents->second.end()); it++) {
			Arc a1(*it, n);
			for (auto it2 = std::next(it); it2 != nParents->second.end(); it2++) {
				if (*it2 != *it) {
					Arc a2(*it2, n);
					headToHeadArcs->emplace(a1, a2);
				}
			}
		}
	}

	return headToHeadArcs;
}

std::shared_ptr<ArcPairSet> DAG::findNotHeadToHeadArcs(NodeId n) {
	auto notHeadToHeadArcs = std::make_shared<ArcPairSet>();

	auto nChildren = m_children.find(n);
	auto nParents = m_parents.find(n);

	if (nChildren != m_children.end()) {
		if (nChildren->second.size() > 1) {
			for (auto it = nChildren->second.begin(); it != std::prev(nChildren->second.end()); it++) {
				Arc a1(n, *it);
				for (auto it2 = std::next(it); it2 != nChildren->second.end(); it2++) {
					if (*it2 != *it) {
						Arc a2(n, *it2);
						notHeadToHeadArcs->emplace(a1, a2);
					}
				}
			}
		}

		if (nParents != m_parents.end()) {
			for (auto it = nParents->second.begin(); it != nParents->second.end(); it++) {
				Arc a1(*it, n);
				for (auto it2 = nChildren->second.begin(); it2 != nChildren->second.end(); it2++) {
					if (*it2 != *it) {
						Arc a2(n, *it2);
						notHeadToHeadArcs->emplace(a1, a2);
					}
				}
			}
		}
	}

	return notHeadToHeadArcs;
}

void DAG::findParentsOfNode(NodeId X, std::shared_ptr<NodeSet> parents) {

	auto Xparents = m_parents.find(X);

	bool checkParents = Xparents != m_parents.end();

	std::vector<NodeId> parentsToCheck;

	if (checkParents) {
		for (auto it = Xparents->second.begin(); it != Xparents->second.end(); it++) {
			addNodesToCheck(*it, parentsToCheck);
			parents->insert(*it);
		}
	}

	if (checkParents) {
		for (int i = 0; i < parentsToCheck.size(); i++) {
			addNodesToCheck(parentsToCheck[i], parentsToCheck);
			parents->insert(parentsToCheck[i]);
		}
	}
}

std::shared_ptr<NodeSet> DAG::findParentsOfNodeset(const NodeSet& L) {
	auto resultNodes = std::make_shared<NodeSet>();

	std::vector<std::future<bool>> resultsFuture;

	for (auto it = L.begin(); it != L.end(); it++) {
		resultNodes->insert(*it);
		findParentsOfNode(*it, resultNodes);
	}

	return resultNodes;
}

//std::shared_ptr<NodeSet> DAG::getNotBarrenVariables(NodeId targetVariable, const std::vector<Evidence>& evidences) {
//	NodeSet nodes(targetVariables);
//
//	for (int i = 0; i < evidences.size(); i++) {
//		nodes.insert(evidences[i].m_id);
//	}
//
//	return findParentsOfNodeset(nodes);
//
//	//if (nodeNotBarren->find(id) != nodeNotBarren->end())
//	//	return false;
//
//	//return true;
//}
