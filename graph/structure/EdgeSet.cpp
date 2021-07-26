#include "EdgeSet.h"

EdgeSet::EdgeSet() {
}

EdgeSet::EdgeSet(const EdgeSet& e) {
	m_edges = e.m_edges;
	m_neighbours = e.m_neighbours;
}

EdgeSet::~EdgeSet() {
}

EdgeSet& EdgeSet::operator=(const EdgeSet& e) {
	if (this != &e) {
		m_edges = e.m_edges;
		m_neighbours = e.m_neighbours;
	}

	return *this;
}

bool EdgeSet::operator==(const EdgeSet& e) const {
	return m_edges == e.m_edges;
}

void EdgeSet::addEdge( NodeId first,  NodeId second) {
	Edge a(first, second);
	if (m_edges.find(a) == m_edges.end()) {
		m_edges.insert(a);

		updateNeighborhood(first, second);
		updateNeighborhood(second, first);
	}
}

void EdgeSet::addEdge(const Edge& edge) {
	if (m_edges.find(edge) == m_edges.end()) {
		m_edges.insert(edge);

		updateNeighborhood(edge.getFirst(), edge.getSecond());
		updateNeighborhood(edge.getSecond(), edge.getFirst());
	}
}

void EdgeSet::addEdges(std::set<Edge> edges) {
	for (auto it = edges.begin(); it != edges.end(); it++) {
		updateNeighborhood(it->getFirst(), it->getSecond());
		updateNeighborhood(it->getSecond(), it->getFirst());
	}
	m_edges.merge(edges);
}

bool EdgeSet::checkEdgeEndpointsInsideNodeSet(const Edge& edge, std::shared_ptr<std::set<NodeId>> nodeSet) {
	return nodeSet->find(edge.getFirst()) != nodeSet->end() && nodeSet->find(edge.getSecond()) != nodeSet->end();
}

bool EdgeSet::checkIfExists(const Edge& edge) const {
	return m_edges.find(edge) != m_edges.end();
}

bool EdgeSet::checkIfIsClique(const std::shared_ptr<NodeSet> nodes) {
	for (auto it = nodes->begin(); it != std::prev(nodes->end()); it++) {
		for (auto it2 = std::next(it); it2 != nodes->end(); it2++) {
			Edge e(*it, *it2);
			if (!checkIfExists(e)) return false;
		}
	}

	return true;
}

std::shared_ptr<std::set<NodeId>> EdgeSet::commonNeighborhood(const NodeId n1, const NodeId n2) {
	auto intersection = std::make_shared<std::set<NodeId>>();
	auto neighborhood1 = m_neighbours.find(n1);
	auto neighborhood2 = m_neighbours.find(n2);

	for (auto it1 = neighborhood1->second.begin(), it2 = neighborhood2->second.begin();;) {
		if (*it1 == *it2) {
			intersection->insert(*it1);
			it1++;
			it2++;
		}

		else if (*it1 < *it2) it1++;
		else it2++;

		if (it1 == neighborhood1->second.end() || it2 == neighborhood2->second.end()) break;
	}

	return intersection;
}

std::shared_ptr<NodeMap> EdgeSet::createNeighborhoodFromEdgeSet(std::shared_ptr<std::set<Edge>> edges) {
	auto neighborhood = std::make_shared<NodeMap>();

	for (auto it = edges->begin(); it != edges->end(); it++) {
		updateNeighborhood(neighborhood, it->getFirst(), it->getSecond());

		updateNeighborhood(neighborhood, it->getSecond(), it->getFirst());
	}

	return neighborhood;
}

std::shared_ptr<std::set<Edge>> EdgeSet::edgeSetDifference(std::shared_ptr<std::set<Edge>> edges1, std::shared_ptr<std::set<Edge>> edges2) {
	auto edgeSetDiff = std::make_shared<std::set<Edge>>(*edges1);

	for (auto it = edges2->begin(); it != edges2->end(); it++) {
		auto edge = edgeSetDiff->find(*it);
		if (edge != edgeSetDiff->end())
			edgeSetDiff->erase(edge);
	}

	return edgeSetDiff;
}

NodeSet EdgeSet::getNodes() {
	NodeSet ns;

	for (auto it = m_neighbours.begin(); it != m_neighbours.end(); it++) {
		ns.insert(it->first);
	}

	return ns;
}

void EdgeSet::removeEdge(NodeId first, NodeId second) {
	Edge e(first, second);

	if (m_edges.find(e) != m_edges.end()) {
		m_edges.erase(e);

		m_neighbours.find(second)->second.erase(first);

		m_neighbours.find(first)->second.erase(second);
	}
}

void EdgeSet::removeEdge(const Edge& e) {
	if (m_edges.find(e) != m_edges.end()) {
		m_edges.erase(e);

		m_neighbours.find(e.getSecond())->second.erase(e.getFirst());

		m_neighbours.find(e.getFirst())->second.erase(e.getSecond());
	}
}

NodeMap::iterator EdgeSet::removeNode(const NodeId n) {
	//for (auto it = m_neighbours.begin(); it != m_neighbours.end(); it++) {
	//	if (it->first != n) {
	//		auto item = it->second.find(n);
	//		if (item != it->second.end())
	//			it->second.erase(item);
	//	}
	//}

	for (auto it = m_neighbours.at(n).begin(); it != m_neighbours.at(n).end(); it++) {
		m_neighbours.at(*it).erase(n);
	}

	return m_neighbours.erase(m_neighbours.find(n));
}

void EdgeSet::updateNeighborhood(NodeId n1, NodeId n2) {
	auto it = m_neighbours.find(n1);
	if (it != m_neighbours.end())
		it->second.insert(n2);
	else {
		NodeSet n;
		n.insert(n2);
		m_neighbours.emplace(n1, n);
	}
}

void EdgeSet::updateNeighborhood(std::shared_ptr<NodeMap> neighborhood, NodeId n1, NodeId n2) {
	auto it = neighborhood->find(n1);
	if (it != neighborhood->end())
		it->second.insert(n2);
	else {
		NodeSet n;
		n.insert(n2);
		neighborhood->emplace(n1, n);
	}
}

std::shared_ptr<std::set<NodeId>> EdgeSet::verticesFromEdges(std::shared_ptr<std::set<Edge>> edges) {
	auto vertices = std::make_shared<std::set<NodeId>>();
	for (auto it = edges->begin(); it != edges->end(); it++) {
		vertices->insert(it->getFirst());
		vertices->insert(it->getSecond());
	}

	return vertices;
}

std::shared_ptr<std::set<NodeId>> EdgeSet::verticesIntersection(std::shared_ptr<std::set<NodeId>> commonNeighborhood, std::shared_ptr<std::set<NodeId>> FVertices) {
	auto verticesInters = std::make_shared<std::set<NodeId>>();

	if (commonNeighborhood->size() == 0) return verticesInters;

	for (auto it1 = commonNeighborhood->begin(), it2 = FVertices->begin();;) {
		if (*it1 == *it2) {
			verticesInters->insert(*it1);
			it1++;
			it2++;
		}
		else if (*it1 < *it2) it1++;
		else it2++;

		if (it1 == commonNeighborhood->end() || it2 == FVertices->end()) break;
	}

	return verticesInters;
}