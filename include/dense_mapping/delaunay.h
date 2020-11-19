//
// Created by chenghe on 6/9/20.
//

#ifndef SUPER_VIO_DELAUNAY_H
#define SUPER_VIO_DELAUNAY_H
#include <utility/eigen_type.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <dense_mapping/triangle.h>
namespace SuperVIO::DenseMapping
{
    class Triangle2D
    {
    public:
        Triangle2D(const cv::Point2f& _a, const cv::Point2f& _b, const cv::Point2f& _c,
                   double _d_a, double _d_b, double _d_c);
        cv::Point2f a;
        cv::Point2f b;
        cv::Point2f c;
        double depth_a;
        double depth_b;
        double depth_c;
    };

    class Delaunay
    {
    public:
        static std::pair<std::vector<cv::Vec3i>, std::vector<cv::Vec2i>>
        Triangulate(const std::vector<cv::Point2f>& points);
    };
}

#endif //SUPER_VIO_DELAUNAY_H
