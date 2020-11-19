//
// Created by chenghe on 6/9/20.
//
#include <dense_mapping/internal/draw_function.h>

namespace SuperVIO::DenseMapping::Internal
{
    /////////////////////////////////////////////////////////////////////////////////////////////
    void applyColorMapLine(const cv::Mat& reference, const cv::Point2f& A,
                           const cv::Point2f& B, cv::Mat* img)
    {
        cv::LineIterator it(*img, A, B);

        for (int ii = 0; ii < it.count; ++ii, ++it)
        {
            cv::Vec3b color = reference.at<cv::Vec3b>(it.pos());
            (*it)[0] = color[0];
            (*it)[1] = color[1];
            (*it)[2] = color[2];
        }
    }
}
