#include "GraphElements.h"

#include <algorithm>

Arc::Arc(NodeId n1, NodeId n2) : m_n1(n1), m_n2(n2) {

}

Arc::Arc(const Arc& a) {
	m_n1 = a.m_n1;
	m_n2 = a.m_n2;
}

Arc::~Arc() {

}

Arc& Arc::operator=(const Arc& a) {
	if (this != &a) {
		m_n1 = a.m_n1;
		m_n2 = a.m_n2;
	}

	return *this;
}

bool Arc::operator==(const Arc& a) const {
	return m_n1 == a.m_n1 && m_n2 == a.m_n2;
}

bool Arc::operator<(const Arc& a) const {
	if (m_n1 < a.m_n1) return true;
	else if (m_n1 == a.m_n1 && m_n2 < a.m_n2) return true;
	else return false;
}

NodeId Arc::getHead() const {
	return m_n1;
}

NodeId Arc::getTail() const {
	return m_n2;
}

NodeId Arc::getOther(NodeId node) const {
	if (node == m_n1) return m_n2;
	else if (node == m_n2) return m_n1;
	else return -1;
}

NodeId Arc::isAdjacent(const Arc& arc) const {
	if (m_n1 == arc.m_n1 || m_n1 == arc.m_n2) return m_n1;
	if (m_n2 == arc.m_n1 || m_n2 == arc.m_n2) return m_n2;
	return -1;
}

bool Arc::isReverse(const Arc& arc) const {
	if (m_n1 == arc.m_n2 && m_n2 == arc.m_n1) return true;
	return false;
}

void Arc::setHead(const NodeId node) {
	if (node != m_n2) m_n1 = node;
}

void Arc::setTail(const NodeId node) {
	if (node != m_n1) m_n2 = node;
}

Edge::Edge(NodeId n1, NodeId n2) : m_n1(std::min(n1, n2)), m_n2(std::max(n1, n2)) {

}

Edge::Edge(const Edge& e) {
	m_n1 = e.m_n1;
	m_n2 = e.m_n2;
}

Edge::~Edge() {

}

Edge& Edge::operator=(const Edge& e) {
	if (this != &e) {
		m_n1 = e.m_n1;
		m_n2 = e.m_n2;
	}
	return *this;
}

bool Edge::operator==(const Edge& e) const {
	return m_n1 == e.m_n1 && m_n2 == e.m_n2;
}

bool Edge::operator<(const Edge& e) const {
	if (m_n1 < e.m_n1) return true;
	else if (m_n1 == e.m_n1 && m_n2 < e.m_n2) return true;
	else return false;
}

NodeId Edge::getFirst() const {
	return m_n1;
}

NodeId Edge::getSecond() const {
	return m_n2;
}

NodeId Edge::getOther(NodeId node) const {
	if (node == m_n1) return m_n2;
	else if (node == m_n2) return m_n1;
	else return -1;
}

void Edge::setFirst(NodeId node) {
	if (node != m_n2) m_n1 = node;
}

void Edge::setSecond(NodeId node) {
	if (node != m_n1) m_n2 = node;
}

std::string Edge::toString() const {
	return std::string(std::to_string(m_n1) + "--" + std::to_string(m_n2));
}