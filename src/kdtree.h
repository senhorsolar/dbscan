#ifndef DBSCAN_KDTREE_H_
#define DBSCAN_KDTREE_H_

#include <iostream>

#include <memory>
#include <numeric>
#include "norms.h"

namespace dbscan
{

using Index=size_t;
using Indices=std::vector<Index>;
    
struct Node
{
    Node(std::unique_ptr<Node>&& left,
	 std::unique_ptr<Node>&& right,
	 Index axis,
	 Index index)
	: left(std::move(left)),
	  right(std::move(right)),
	  axis(axis),
	  index(index)
	{}
    
    std::unique_ptr<Node> left;  ///< Left node
    std::unique_ptr<Node> right; ///< Right node
    Index axis; ///< Splitting plane
    Index index; ///< Index of point in data for this node
};    

template <class Matrix>
class KDTree
{
public:
        
    /**
     * Assume data is a 2D array
     *
     * NOTE: This version of KDTree is dangerous, since it assumes the lifetime
     * of KDTree does not outlive the lifetime of data. This is okay for the
     * provided implementation of DBScan.
     */
    KDTree(const Matrix& data);

    template <class Point, class DistFunc>
    Indices RangeQuery(Point p, double radius, DistFunc dist_func);

    template <class Point>
    Indices RangeQuery(Point p, double radius) {
	return RangeQuery(p, radius, norms::Euclidean<Point>);
    }
    
private:
            
    std::unique_ptr<Node> BuildTree(const Matrix& data);

    const Matrix& m_data; ///< Reference to original data
    std::unique_ptr<Node> m_root;  ///< Root of KDTree
};


template <class Matrix>
KDTree<Matrix>::KDTree(const Matrix& data)
    : m_data(data),
      m_root(BuildTree(data))
{    
}

template <class Matrix>
std::unique_ptr<Node>
KDTree<Matrix>::BuildTree(const Matrix& data)
{
    if (data.size() == 0 || data[0].size() == 0)
	return nullptr;

    size_t ndims = data[0].size();

    // Indices of data
    Indices idxs_remaining(data.size());
    std::iota(idxs_remaining.begin(), idxs_remaining.end(), 0);
    
    // Recursive helper function to build tree
    // Note: defining type so that a lambda can be used recursively
    std::function<std::unique_ptr<Node>(Index, Index, Index)> _builder;    
    _builder = [&](Index lo, Index hi, Index depth) -> std::unique_ptr<Node>
    {
	if (lo >= hi)
	    return nullptr;

	Index axis = depth % ndims;
	
	// Determine splitting plane
	// -------------------------
	// Put the median in the correct location, such that
	//   data[idx][axis] < data[mid][axis] for idx in idxs_remaining[1...mid-1]
	// and
	//   data[idx][axis] > data[mid][axis] for idx in idxs_remaining[mid+1...]
	size_t mid = lo + (hi - lo) / 2;  /* where median should be */
	nth_element(idxs_remaining.begin() + lo,
		    idxs_remaining.begin() + mid,
		    idxs_remaining.begin() + hi,
		    [&](Index l, Index r) {
			return data[l][axis] <= data[r][axis];
		    });

	return std::make_unique<Node>(_builder(lo, mid, depth + 1),
				      _builder(mid + 1, hi, depth + 1),
				      axis,
				      idxs_remaining[mid]);
	
    };
       
    return _builder(0, data.size(), 0);
}


template <class Matrix>
template <class Point, class DistFunc>
Indices KDTree<Matrix>::RangeQuery(Point p, double radius, DistFunc dist_func)
{
    Indices point_idxs;

    std::function<void(const std::unique_ptr<Node>&)> _range_query;
    _range_query = [&](const std::unique_ptr<Node>& node)
    {
	if (node) {

	    Index idx = node->index;
	    Index axis = node->axis;

	    auto dist_to_node = dist_func(m_data[idx], p);
	    auto dist_to_splitting_plane = std::abs(m_data[idx][axis] - p[axis]);
	    
	    if (dist_to_node <= radius)
		point_idxs.push_back(idx);	    	    

	    if (p[axis] <= m_data[idx][axis]) {
		_range_query(node->left);
		if (dist_to_splitting_plane <= radius) {
		    _range_query(node->right);
		}
	    }
	    else {
		_range_query(node->right);
		if (dist_to_splitting_plane <= radius) {
		    _range_query(node->left);
		}
	    }		   	    	    
	}
    };
    
    _range_query(m_root);
    return point_idxs;
}



} // namespace dbscan

#endif // DBSCAN_KDTREE_H_
