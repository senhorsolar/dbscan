#ifndef DBSCAN_H_
#define DBSCAN_H_

#include <stdexcept>
#include <vector>
#include "kdtree.h"
#include "norms.h"

namespace dbscan
{

constexpr ssize_t NOISE = -1; ///< Outlier
constexpr ssize_t UNDEFINED = -2; ///< Unlabeled point
constexpr ssize_t IN_PROGRESS = -3;  ///< UNDEFINED but on queue
    
template <class Matrix>
std::vector<ssize_t> DBSCAN(const Matrix& data, double eps, int minpts)
{
    return DBSCAN(data, eps ,minpts,
		  [](const auto& v1, const auto& v2) {
		      return norms::Euclidean(v1, v2);
		  });
}

template <class Matrix, class DistFunc>
std::vector<ssize_t> DBSCAN(const Matrix& data, double eps, int minpts, DistFunc dist_func)
{

    // Does not copy data
    KDTree<Matrix> kdtree(data);

    std::vector<ssize_t> labels(data.size(), UNDEFINED);

    // Current cluster label
    ssize_t cluster_label = 0;

    
    for (size_t i = 0; i < data.size(); ++i) {  
	
	// Already visited this node
	if (labels[i] != UNDEFINED) {
	    continue;
	}

	auto neighbor_idxs = kdtree.RangeQuery(data[i], eps, dist_func);

	// Check if point is an outlier
	if (neighbor_idxs.size() < minpts) {
	    labels[i] = NOISE;
	    continue;
	}

	labels[i] = cluster_label;
	    
	for (auto j : neighbor_idxs) {	    
	    if (labels[j] == NOISE)
		labels[j] = cluster_label;
	    else if (labels[j] == UNDEFINED) {
		labels[j] = IN_PROGRESS;
	    }
	}

	while (!neighbor_idxs.empty()) {

	    auto q = neighbor_idxs.back();
	    neighbor_idxs.pop_back();
	    
	    if (labels[q] != IN_PROGRESS) {
		continue;
	    }

	    labels[q] = cluster_label;
	    
	    auto other_neighbors = kdtree.RangeQuery(data[q], eps, dist_func);	    

	    if (other_neighbors.size() < minpts) {
		continue;
	    }

	    // Filter for all undefined labels
	    for (auto j : other_neighbors) {
		if (labels[j] == NOISE) {
		    labels[j] = cluster_label;
		}
		else if (labels[j] == UNDEFINED) {
		    labels[j] = IN_PROGRESS;
		    neighbor_idxs.push_back(j);
		}
	    }
	}

	// Check for overflow
	if (cluster_label + 1 < cluster_label) {
	    // Check if we still have points to check
	    if (i < (data.size() - 1)) {
		throw std::overflow_error("Surpassed maximum number of clusters. Try increasing radius.");
	    }
	}
	++cluster_label;
    }

    return labels;
}
    
} // namespace dbscan

#endif // DBSCAN_H_
