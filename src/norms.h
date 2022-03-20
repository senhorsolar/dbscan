#ifndef DBSCAN_NORMS_H_
#define DBSCAN_NORMS_H_

#include <cmath>

namespace dbscan {
namespace norms {

// Finds the euclidean distance between two vectors
// NOTE: Assumes vectors are the same size
template <class Vector>
double Euclidean(Vector v1, Vector v2)
{ 
    double dist = 0;
    for (int i = 0; i < v1.size(); ++i) {
	dist += pow(v1[i] - v2[i], 2);
    }
    return sqrt(dist);
}

} // namespace norms
} // namespace dbscan

#endif // DBSCAN_NORMS_H
