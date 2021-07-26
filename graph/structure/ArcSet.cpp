#include "ArcSet.h"

ArcSet::ArcSet() : m_numberOfNodes(0){}

ArcSet::ArcSet(const ArcSet& a) {
	m_numberOfNodes = a.m_numberOfNodes;
	m_arcs = a.m_arcs;
	m_children = a.m_children;
	m_parents = a.m_parents;
}

ArcSet::~ArcSet() {
}

ArcSet& ArcSet::operator=(const ArcSet& a) {
	if (this != &a) {
		m_numberOfNodes = a.m_numberOfNodes;
		m_arcs = a.m_arcs;
		m_parents = a.m_parents;
		m_children = a.m_children;
	}

	return *this;
}

bool ArcSet::operator==(const ArcSet& a) const {
	return m_arcs == a.m_arcs;
}

void ArcSet::addArc(const NodeId head, const NodeId tail) {
	Arc a(head, tail);
	if (m_arcs.find(a) == m_arcs.end()) {
		m_arcs.insert(a);

		auto it = m_parents.find(tail);
		if (it != m_parents.end())
			it->second.insert(head);
		else {
			NodeSet n;
			n.insert(head);
			m_parents.emplace(tail, n);
			m_numberOfNodes++;
		}

		auto it2 = m_children.find(head);
		if (it2 != m_children.end())
			it2->second.insert(tail);
		else {
			NodeSet n;
			n.insert(tail);
			m_children.emplace(head, n);
		}
	}	
}

void ArcSet::addArc( Arc& arc) {
	if (m_arcs.find(arc) == m_arcs.end())
		m_arcs.insert(arc);

	auto it = m_parents.find(arc.getTail());
	if (it != m_parents.end())
		it->second.insert(arc.getHead());
	else {
		NodeSet n;
		n.insert(arc.getHead());
		m_parents.emplace(arc.getTail(), n);
	}

	auto it2 = m_children.find(arc.getHead());
	if (it2 != m_children.end())
		it2->second.insert(arc.getTail());
	else {
		NodeSet n;
		n.insert(arc.getTail());
		m_children.emplace(arc.getHead(), n);
	}
}

bool ArcSet::checkIfExists(const Arc& arc) {
	if (m_arcs.find(arc) != m_arcs.end()) return true;
	else return false;
}

NodeSet ArcSet::getNodes() {
	NodeSet ns;

	for (auto it = m_children.begin(); it != m_children.end(); it++) {
		ns.insert(it->first);
	}

	for (auto it = m_parents.begin(); it != m_parents.end(); it++) {
		ns.insert(it->first);
	}

	return ns;
}

void ArcSet::removeArc(const NodeId head, const NodeId tail) {
	Arc a(head, tail);

	if (m_arcs.find(a) != m_arcs.end()) {
		m_arcs.erase(a);

		m_parents.find(tail)->second.erase(head);

		m_children.find(head)->second.erase(tail);

	}
}

void ArcSet::removeArc(const Arc& arc) {

	auto it = m_arcs.find(arc);
	if (it != m_arcs.end()) {
		m_arcs.erase(it);

		m_parents.find(arc.getTail())->second.erase(arc.getHead());

		m_children.find(arc.getHead())->second.erase(arc.getTail());
	}
}

void ArcSet::setNumberOfNodes(int num) {
	m_numberOfNodes = num;
}