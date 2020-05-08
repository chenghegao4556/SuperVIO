//
// Created by chenghe on 3/11/20.
//

#ifndef SUPER_VIO_TRIANGULATOR_H
#define SUPER_VIO_TRIANGULATOR_H

#include <opencv2/core/core.hpp>
#include <utility/eigen_type.h>
#include <utility/eigen_base.h>
#include <vision/camera.h>

namespace SuperVIO::Vision
{
    class Triangulator
    {
    public:
        typedef Camera::ConstPtr CameraConstPtr;

        enum class Status
        {
            NotGoodCandidate = 0,
            LargeReprojectionError = 1,
            NegativeDepth = 2,
            Success = 3,
        };

        class Parameters
        {
        public:
            explicit Parameters(bool _baseline_check = false,
                                bool _view_angle_check = false,
                                double _baseline_thresh = 0.0,
                                double _view_angle_thresh = 0.0,
                                double _error_thresh = 5.0);

            bool baseline_check;
            bool view_angle_check;
            double baseline_thresh;
            double view_angle_thresh;
            double error_thresh;
        };

        class Result
        {
        public:
            explicit Result(Status _status = Status::NotGoodCandidate,
                            double _error = 0,
                            Vector3 _world_point = {0, 0, 0},
                            double _depth = -1);

            Status status;
            double mean_reprojection_error;
            Vector3 world_point;
            double depth;
        };

        /**
         * @brief triangulate points that belond to same feature
         * @param[in] positions (t_w_i)
         * @param[in] rotations  (q_w_i)
         * @param[in] image_points (key points)
         * @param[in] r_i_c (extrinsic parameters between imu and camera)
         * @param[in] t_i_c
         * @param[in] camera_ptr
         * @param[in] parameters
         */
        static Result TriangulatePoints(const Vector3s& positions,
                                        const Quaternions& rotations,
                                        const std::vector<cv::Point2f>& image_points,
                                        const Quaternion& r_i_c,
                                        const Vector3& t_i_c,
                                        const CameraConstPtr& camera_ptr,
                                        const Parameters& parameters);

    protected:

        /**
         * @brief using svd method
         */
        static std::pair<Vector3, double>
        TriangulateDLT(const Pose3s& poses, const Vector3s& points);

        static bool GoodMeasurements(const Pose3s& poses,
                                     const Vector2s& points,
                                     const CameraConstPtr& camera_ptr,
                                     const Parameters& parameters);

        /**
         * @brief check measurements are valid for lindstrom method
         */
        static bool IsGoodCandidate(const Pose3& pose0, const Pose3& pose1,
                                    const Vector2& point0, const Vector2& point1,
                                    const CameraConstPtr& camera_ptr,
                                    const Parameters& parameters);
    };
}

#endif //SUPER_VIO_TRIANGULATOR_H
