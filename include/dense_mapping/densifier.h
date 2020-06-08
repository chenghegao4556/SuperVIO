//
// Created by chenghe on 5/12/20.
//

#ifndef SUPER_VIO_DENSIFIER_H
#define SUPER_VIO_DENSIFIER_H
#include <ros/ros.h>
#include <ros/console.h>
#include <utility/eigen_type.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <dense_mapping/fast_bilateral_solver.h>
namespace SuperVIO::DenseMapping
{
    class Densifier
    {
    public:
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

        static cv::Mat
        Evaluate(const cv::Mat& image, const std::vector<cv::Point2f>& points,
                 const std::vector<double>& depth);
    protected:
        static std::vector<Triangle2D>
        DelaunayTriangulate(const cv::Size& image_size, const std::vector<cv::Point2f>& points,
                            const std::vector<double>& depths);

        static std::pair<cv::Mat, cv::Mat>
        InterpolateMesh(const cv::Size& image_size, const std::vector<Triangle2D>& triangles);

        static cv::Mat
        OptimizeDepthMap(const cv::Mat& image, const cv::Mat& depth_map, const cv::Mat& confidence_map);
    };//end of Densifier
}//end of SuperVIO::DenseMapping

#endif //SUPER_VIO_DENSIFIER_H
