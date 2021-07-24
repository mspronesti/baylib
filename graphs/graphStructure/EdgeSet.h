#pragma once

#include "GraphElements.h"

#include <memory>

typedef std::set<NodeId> NodeSet;
typedef std::map<NodeId, NodeSet> NodeMap;

class EdgeSet {
public:

	//constructor
	EdgeSet();

	//copy constructor
	EdgeSet(const EdgeSet& a);

	//destructor
	~EdgeSet();

	//copy operator
	EdgeSet& operator=(const EdgeSet& e);

	//boolean operator
	bool operator==(const EdgeSet& e) const;

	void addEdge(const NodeId head, const NodeId tail);

	void addEdge(const Edge& e);

	bool checkIfExists(const Edge& e) const;

	//returns the NodeSet containing all nodes in the old_graph
	NodeSet getNodes();

	//returns the number of nodes in the old_graph
	inline int getNumOfVertices() { return m_neighbours.size(); };

	void removeEdge(const NodeId head, const NodeId tail);

	void removeEdge(const Edge& e);

protected:

	void addEdges(std::set<Edge> edges);

	//checks whether both head and tail of an edge are inside a NodeSet, returns true if yes, false otherwise
	bool checkEdgeEndpointsInsideNodeSet(const Edge& edge, std::shared_ptr<NodeSet> nodeSet);

	//checks if the Nodeset is forms a clique, returns true if yes, false otherwise
	bool checkIfIsClique(const std::shared_ptr<NodeSet> nodes);

	//returns the neighborhood formed by the edges
	std::shared_ptr<NodeMap> createNeighborhoodFromEdgeSet(std::shared_ptr<std::set<Edge>> edges);

	//returns the intersection of 2 nodes' neighborhood
	std::shared_ptr<NodeSet> commonNeighborhood(const NodeId n1, const NodeId n2);

	//returns the difference between 2 sets of edges
	std::shared_ptr<std::set<Edge>> edgeSetDifference(std::shared_ptr<std::set<Edge>> edges1, std::shared_ptr<std::set<Edge>> edges2);

	//removes a node from the old_graph, returns the iterator to the next element in the map after deletion
	NodeMap::iterator removeNode(const NodeId id);

	//inserts node n2 in node n1 neighborhood
	void updateNeighborhood(const NodeId n1, const NodeId n2);

	//given a neighborhood, inserts inserts node n2 in node n1 neighborhood
	void updateNeighborhood(std::shared_ptr<NodeMap> neighborhood, const NodeId n1, const NodeId n2);

	//returns the NodeSet obtained from a set of edges
	std::shared_ptr<NodeSet> verticesFromEdges(const std::shared_ptr<std::set<Edge>> edges);

	//returns the intersection of 2 NodeSet
	std::shared_ptr<NodeSet> verticesIntersection(const std::shared_ptr<NodeSet> commonNeighborhood, const std::shared_ptr<NodeSet> FVertices);

	std::set<Edge> m_edges;
	NodeMap m_neighbours;
};