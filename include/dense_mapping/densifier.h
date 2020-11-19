//
// Created by chenghe on 5/12/20.
//

#ifndef SUPER_VIO_DENSIFIER_H
#define SUPER_VIO_DENSIFIER_H
#include <ros/ros.h>
#include <ros/console.h>
#include <utility/eigen_type.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <dense_mapping/delaunay.h>
#include <dense_mapping/fast_bilateral_solver.h>
#include <dense_mapping/internal/draw_function.h>
#include <dense_mapping/mesh_regularizer.h>
namespace SuperVIO::DenseMapping
{
    class Densifier
    {
    public:

        static cv::Mat
        Evaluate(const cv::Mat& image, const std::vector<cv::Point2f>& points,
                 const std::vector<double>& depth);

        static cv::Mat
        VisualizeDepthMap(const cv::Mat& image, const cv::Mat& raw_depth_map, const cv::Mat& fine_depth_map,
                          const std::vector<Triangle2D>& triangles);
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
