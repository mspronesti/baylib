#pragma once

#include <mutex>

#include "graphStructure/ArcSet.h"
#include "../tools/thread_pool.hpp"

typedef std::set<std::pair<Arc, Arc>> ArcPairSet;

struct NodeSeparated {
	NodeId m_id;
	bool m_separated;

	NodeSeparated() : m_id(-1), m_separated(false) {};

	NodeSeparated(NodeId id) : m_id(id), m_separated(false) {};

	NodeSeparated(NodeId id, bool separated) : m_id(id), m_separated(separated) {};

	bool operator<(const NodeSeparated& ns) const {
		return m_id < ns.m_id;
	}
};

class DAG : public ArcSet {
public:
	//constructor
	DAG();

	//copy constructor
	DAG(const DAG& d);

	//destructor
	~DAG();


private:
	//adds the parents of node n to the list of nodes to be checked
	void addNodesToCheck(NodeId n, std::vector<NodeId>& nodeSetToCheck);

	//adds reverse arc for each arc to the old_graph
	std::shared_ptr<DAG> createReverseDAG();

	//returns the set of dConnected nodes to NodeSet J
	void dConnectedNodes(NodeSet& J, std::shared_ptr<ArcPairSet> legalArcpairs, NodeSet& R);

	//returns the set of head-to-head arcs where node n is the node in common
	std::shared_ptr<ArcPairSet> findHeadToHeadArcs(NodeId n);

	//returns the set of legal pairs of arcs given a NodeSet L (the set of evidences introduced) and the set of nodes that have at keast a descendent in L
	std::shared_ptr<ArcPairSet> findLegalPairsOfArcs(NodeSet& L, std::shared_ptr<NodeSet> descendents);

	//returns the set of not head-to-head arcs where node n is the node in common
	std::shared_ptr<ArcPairSet> findNotHeadToHeadArcs(NodeId n);

	//finds all the parents of node X and puts them into parents
	void findParentsOfNode(NodeId X, std::shared_ptr<NodeSet> parents);

	//returns all the parents of the NodeSet L
	std::shared_ptr<NodeSet> findParentsOfNodeset(const NodeSet& L);


	std::shared_ptr<DAG> m_reverseDAG;

	std::once_flag m_reverseDAGCreated;
	std::mutex m_headToHeadMutex;
	std::mutex m_notHeadToHeadMutex;
	std::mutex m_evidenceNodesMutex;
};