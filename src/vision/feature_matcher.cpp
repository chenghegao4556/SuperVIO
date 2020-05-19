//
// Created by chenghe on 3/10/20.
//
#include <vision/feature_matcher.h>
#include <vision/feature_extractor.h>

namespace SuperVIO::Vision
{

    typedef FeatureMatcher::Matches Matches;

    /////////////////////////////////////////////////////////////////////////////////////////////
    std::pair<double, Matches> FeatureMatcher::
    Match(const FrameMeasurement& train_frame,
          const FrameMeasurement& query_frame,
          float match_thresh,
          float ratio_thresh,
          bool  f_check)
    {
        std::vector<std::vector<cv::DMatch>> matches;
        std::map<int, cv::DMatch> match_map;
        std::vector<cv::DMatch> best_matches;

        auto matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_L2);

        cv::cuda::GpuMat gpu_train_descriptors, gpu_query_descriptors;
        gpu_train_descriptors.upload(train_frame.descriptors);
        gpu_query_descriptors.upload(query_frame.descriptors);

        matcher->knnMatch(gpu_query_descriptors, gpu_train_descriptors, matches, 2);

        for(auto& match_pair: matches)
        {
            const auto& best_match        = match_pair[0];
            const auto& second_best_match = match_pair[1];
            if(best_match.distance < match_thresh)
            {
                const float ratio = best_match.distance / second_best_match.distance;
                if(ratio < ratio_thresh)
                {
                    auto iter = match_map.find(best_match.trainIdx);
                    if(iter == match_map.end())
                    {
                        match_map[best_match.trainIdx] = best_match;
                    }
                    else
                    {
                        match_map.erase(best_match.trainIdx);
                    }
                }
            }
        }

        for(const auto& match: match_map)
        {
            best_matches.push_back(match.second);
        }

        if(f_check && best_matches.size() >= 8)
        {
            best_matches = FCheck(train_frame, query_frame, best_matches);
        }

        double parallax = ComputeParallax(train_frame, query_frame, best_matches);

        return std::make_pair(parallax, best_matches);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    std::vector<cv::DMatch> FeatureMatcher::
    FCheck(const FrameMeasurement& train_frame,
           const FrameMeasurement& query_frame,
           const Matches& matches)
    {

        const auto& match_count = matches.size();
        cv::Mat train_points(match_count, 2, CV_32F);
        cv::Mat query_points(match_count, 2, CV_32F);

        for (size_t i = 0; i < match_count; ++i)
        {
            const auto train_index = matches[i].trainIdx;
            const auto query_index = matches[i].queryIdx;

            const auto& train_point = train_frame.key_points[train_index].point;
            const auto& query_point = query_frame.key_points[query_index].point;

            train_points.at<float>(i, 0) = train_point.x;
            train_points.at<float>(i, 1) = train_point.y;

            query_points.at<float>(i, 0) = query_point.x;
            query_points.at<float>(i, 1) = query_point.y;
        }
        std::vector<uchar> inliers;
        cv::findFundamentalMat(train_points, query_points, inliers, cv::FM_RANSAC, 2, 0.99);
        std::vector<cv::DMatch> best_matches;
        for(size_t i = 0; i < match_count; ++i)
        {
            if(inliers[i])
            {
                best_matches.push_back(matches[i]);
            }
        }

        return best_matches;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    double FeatureMatcher::
    ComputeParallax(const FrameMeasurement& train_frame,
                    const FrameMeasurement& query_frame,
                    const Matches& matches)
    {
        double parallax = 0;
        for (const auto& match : matches)
        {
            const auto& train_index = match.trainIdx;
            const auto& query_index = match.queryIdx;

            const auto& train_point = train_frame.key_points[train_index].point;
            const auto& query_point = query_frame.key_points[query_index].point;

            const float diff_x = train_point.x - query_point.x;
            const float diff_y = train_point.y - query_point.y;

            parallax += std::sqrt(diff_x * diff_x + diff_y * diff_y);
        }

        parallax /= static_cast<double>(matches.size());

        return parallax;
    }

}//end of SuperVIO