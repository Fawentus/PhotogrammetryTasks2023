#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    // параметры для приближенного поиска
    index_params = flannKdTreeIndexParams(2); // My param
    search_params = flannKsTreeSearchParams(60); // My param
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    matches.clear();
    cv::Mat indices, dists;
    flann_index->knnSearch(query_desc, indices, dists, k, *search_params);

    for (int i = 0; i < query_desc.rows; i++) {
        matches.emplace_back();
        matches[i].reserve(k);
        for (int j = 0; j < k; j++) {
            cv::DMatch match = cv::DMatch(i, indices.at<int>(i, j), dists.at<float>(i, j));
            matches[i].push_back(match);
        }
    }
}
