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
    class Delaunay
    {
    public:
        static std::vector<cv::Vec3i>
        Triangulate(const std::vector<cv::Point2f>& points);
    };
}

#endif //SUPER_VIO_DELAUNAY_H
