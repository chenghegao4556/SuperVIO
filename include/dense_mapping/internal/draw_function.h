//
// Created by chenghe on 6/9/20.
//

#ifndef SUPER_VIO_DRAW_FUNCTION_H
#define SUPER_VIO_DRAW_FUNCTION_H
#include <memory>
#include <opencv2/imgproc/imgproc.hpp>
namespace SuperVIO::DenseMapping::Internal
{
    void applyColorMapLine(const cv::Mat& reference, const cv::Point2f& A,
                           const cv::Point2f& B, cv::Mat* img);
}

#endif //SRC_DRAW_FUNCTION_H
