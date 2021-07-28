#pragma once

#include "GraphElements.h"

#include <memory>

typedef std::set<NodeId> NodeSet;
typedef std::map<NodeId, NodeSet> NodeMap;

class ArcSet {
public:

	//constructor
	ArcSet();

	//copy constructor
	ArcSet(const ArcSet& a);

	//destructor
	~ArcSet();

	//copy operator
	ArcSet& operator=(const ArcSet& a);

	//boolean operator
	bool operator==(const ArcSet& a) const;

	void addArc(const NodeId head, const NodeId tail);

	void addArc(Arc& arc);

	bool checkIfExists(const Arc& arc);

	//returns the NodeSet containing all nodes in the old_graph
	NodeSet getNodes();

	//returns the number of nodes in the old_graph
	inline int getNumberOfNodes() { return m_numberOfNodes; };

	inline bool hasChildren(const NodeId node) { return m_children.find(node) != m_children.end(); };

	void removeArc(const NodeId head, const NodeId tail);

	void removeArc(const Arc& arc);

	void setNumberOfNodes(int num);

protected:
	int m_numberOfNodes;
	std::set<Arc> m_arcs;

	//a map containing for each node the set of its parents
	NodeMap m_parents;

	//a map containing for each node the set of its children
	NodeMap m_children;
};