//
// Created by chenghe on 6/26/20.
//
#include <vision/feature_extractor.h>
#include <vision/feature_tracker.h>
using namespace SuperVIO;
int main()
{
    std::string weight_path("/home/chenghe/catkin_ws/src/SuperVIO/data/superpoint.pt");
    Vision::FeatureExtractor::Parameters parameters(10, 10, cv::Size(640, 480), 8, 0.015f);
    auto feature_extractor_ptr = Vision::FeatureExtractor::Creat(parameters, weight_path);

    auto image0 = cv::imread("1.jpg", 0);
    auto image1 = cv::imread("2.jpg", 0);

    auto frame_measurement0 = feature_extractor_ptr->Compute(image0);
    auto frame_measurement1 = feature_extractor_ptr->Compute(image1);

    auto matches_01 = Vision::FeatureMatcher::Match(frame_measurement0, frame_measurement1,
                                                    0.8, 0.8, false).second;

    std::vector<cv::KeyPoint> k1;
    std::vector<cv::KeyPoint> k2;
    std::for_each(frame_measurement0.key_points.begin(), frame_measurement0.key_points.end(),
                  [&](const SuperVIO::Vision::KeyPoint& k)
                  {
                      k1.emplace_back(k.point, 0);
                  });

    std::for_each(frame_measurement1.key_points.begin(), frame_measurement1.key_points.end(),
                  [&](const SuperVIO::Vision::KeyPoint& k)
                  {
                      k2.emplace_back(k.point, 0);
                  });
    std::cout<<" "<<k1.size()<<" "<<k2.size()<<std::endl;

    cv::Mat out;
    cv::drawMatches(image1, k2, image0, k1, matches_01, out);
    cv::imwrite("out.jpg", out);
    return 0;

}
