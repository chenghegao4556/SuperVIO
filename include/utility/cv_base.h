//
// Created by chenghe on 3/24/20.
//

#ifndef SUPER_VIO_CV_BASE_H
#define SUPER_VIO_CV_BASE_H

#include <utility/eigen_base.h>
#include <opencv2/core.hpp>
namespace SuperVIO::Utility
{
    class CVBase
    {
    public:
        static Vector2 Point2fToVector2(const cv::Point2f& pt)
        {
            return Vector2{pt.x, pt.y};
        }
    };
}

#endif //SUPER_VIO_CV_BASE_H
