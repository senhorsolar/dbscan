#include <iostream>
#include <array>
#include <vector>
#include "../src/dbscan.h"

using Point=std::array<int, 2>;
using Matrix=std::vector<Point>;

int main()
{
    Matrix m = {{0, 0},
		{0, -1},
		{0, 1},
		{3, 0},
		{3, -1},
		{3, 1},
		{6, 0},
		{6, -1},
		{6, 1}};

    // dbscan::KDTree<Matrix> kdtree(m);

    // auto neighbors = kdtree.RangeQuery(Point({0, 0}), 2);

    // for (auto& idx : neighbors) {
    // 	std::cout << "idx: " << idx << '\n';
    // }
    
    auto cluster_labels = dbscan::DBSCAN(m, 2, 3);

    for (auto& label : cluster_labels) {
    	std::cout << "label: " << label << '\n';
    }
    
    return 0;
};
