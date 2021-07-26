#pragma once

#include <vector>
#include <set>
#include <map>
#include <string>

typedef int NodeId;

//directed edges of the old_graph
class Arc {
public:
	//constructor
	Arc(NodeId n1, NodeId n2);

	//copy constructor
	Arc(const Arc& a);

	//destructor
	~Arc();

	//copy operator
	Arc& operator=(const Arc& a);

	//boolean operators
	bool operator==(const Arc& a) const;

	bool operator<(const Arc& a) const;

	//returns NodeId of arc's head
	NodeId getHead() const;

	//returns NodeId of arc's tail
	NodeId getTail() const;

	//returns NodeId of arc's other extremity
	NodeId getOther(NodeId node) const;

	//checks whether 2 arcs are adjacent, if yes returns the NodeId of the common node, else returns -1
	NodeId isAdjacent(const Arc& arc) const;

	//checks whether 2 arcsa are one the reverse of the other
	bool isReverse(const Arc& arc) const;

	//changes the arc's head
	void setHead(const NodeId node);

	//changes the arc's tail
	void setTail(const NodeId node);

private:
	NodeId m_n1, m_n2;
};

//undirected edges of the old_graph
class Edge {
public:
	//constructor
	Edge(NodeId n1, NodeId n2);

	//copy constructor
	Edge(const Edge& e);

	//destructor
	~Edge();

	//copy operator
	Edge& operator=(const Edge& e);

	//boolean operators
	bool operator==(const Edge& e) const;

	bool operator<(const Edge& e) const;

	//return NodeId of edge's first node
	NodeId getFirst() const;

	//return NodeId of edge's second node
	NodeId getSecond() const;

	//return NodeId of edge's other extremity
	NodeId getOther(NodeId node) const ;

	//changes the edge's first node
	void setFirst(NodeId node);

	//changes the edge's second node
	void setSecond(NodeId node);

	std::string toString() const ;

private:
	NodeId m_n1, m_n2;
};