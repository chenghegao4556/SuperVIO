//
// Created by chenghe on 3/11/20.
//
#include <vision/triangulator.h>

namespace SuperVIO::Vision
{
    /////////////////////////////////////////////////////////////////////////////////////
    Triangulator::Result::
    Result(Status _status, double _error, Vector3 _world_point, double _depth):
            status(_status),
            mean_reprojection_error(_error),
            world_point(std::move(_world_point)),
            depth(_depth)
    {

    }

    /////////////////////////////////////////////////////////////////////////////////////
    Triangulator::Parameters::
    Parameters(bool _baseline_check, bool _view_angle_check,
               double _baseline_thresh, double _view_angle_thresh,
               double _error_thresh):
            baseline_check(_baseline_check),
            view_angle_check(_view_angle_check),
            baseline_thresh(_baseline_thresh),
            view_angle_thresh(_view_angle_thresh),
            error_thresh(_error_thresh)
    {

    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    Triangulator::Result Triangulator::
    TriangulatePoints(const Vector3s& positions,
                      const Quaternions& rotations,
                      const std::vector<cv::Point2f>& image_points,
                      const Quaternion& r_i_c,
                      const Vector3& t_i_c,
                      const CameraConstPtr& camera_ptr,
                      const Parameters& parameters)
    {
        if(positions.size() == rotations.size() &&
           positions.size() == image_points.size() &&
           positions.size() >= 2 &&
           camera_ptr != nullptr)
        {
            Pose3s poses;
            Vector3s h_points;
            Vector2s i_points;
            for(size_t i = 0; i < positions.size(); ++i)
            {

                Quaternion r_w_c = rotations[i] * r_i_c;
                Vector3 p_w_c = positions[i] + rotations[i] * t_i_c;

                Pose3 T_w_c = Utility::EigenBase::ToPose3(r_w_c, p_w_c);
                poses.push_back(T_w_c);

                i_points.emplace_back(static_cast<double>(image_points[i].x),
                                      static_cast<double>(image_points[i].y));
                Vector3 pt_h = camera_ptr->BackProject(i_points.back());
                h_points.push_back(pt_h);
            }
            Result result;

            if(GoodMeasurements(poses, i_points, camera_ptr, parameters))
            {
                auto dlt_result = TriangulateDLT(poses, h_points);
                result.world_point = dlt_result.first;
                result.depth       = dlt_result.second;
            }
            else
            {
                return Result(Status::NotGoodCandidate);
            }


            double error = 0;
            for(size_t i = 0; i < poses.size(); ++i)
            {
                auto c_point = poses[i].inverse() * result.world_point;

                if(c_point.z() <= 0)
                {
                    result.status = Status::NegativeDepth;

                    return result;
                }

                error += (camera_ptr->Project(c_point) - i_points[i]).norm();
            }

            result.mean_reprojection_error = error / static_cast<double>(h_points.size());
            if(result.mean_reprojection_error > parameters.error_thresh)
            {
                result.status = Status::LargeReprojectionError;
            }
            else
            {
                result.status = Status::Success;
            }

            return result;
        }
        else
        {
            return Result();
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    std::pair<Vector3, double>  Triangulator::
    TriangulateDLT(const Pose3s& poses, const Vector3s& points)
    {
        const auto& T_w_c0 = poses[0];
        MatrixX A(2 * poses.size(), 4);
        for(size_t i = 0; i < poses.size(); ++i)
        {
            size_t index = 2 * i;
            const auto& T_w_ci  = poses[i];
            const Pose3 T_ci_c0 = T_w_ci.inverse() * T_w_c0;
            A.row(index)     = points[i].x() * T_ci_c0.matrix().row(2) - T_ci_c0.matrix().row(0);
            A.row(index + 1) = points[i].y() * T_ci_c0.matrix().row(2) - T_ci_c0.matrix().row(1);
        }

        Vector4 V = Eigen::JacobiSVD<MatrixX>(A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        const Vector3 camera_point_0 = V.head(3)/V(3);
        const double  depth = camera_point_0.z();
        const Vector3 world_point = T_w_c0 * camera_point_0;

        return std::make_pair(world_point, depth);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    bool Triangulator::
    GoodMeasurements(const Pose3s& poses,
                     const Vector2s& points,
                     const CameraConstPtr& camera_ptr,
                     const Parameters& parameters)
    {
        bool state = false;
        for (size_t i = 0; i < poses.size(); ++i)
        {
            for(size_t j = i+1; j < poses.size(); ++j)
            {
                state = IsGoodCandidate(poses[i], poses[j], points[i], points[j], camera_ptr, parameters);

                if(state)
                {
                    return state;
                }

            }
        }

        return state;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    bool Triangulator::
    IsGoodCandidate(const Pose3& pose0, const Pose3& pose1,
                    const Vector2& point0, const Vector2& point1,
                    const CameraConstPtr& camera_ptr,
                    const Parameters& parameters)
    {
        Vector3 view0 = point0.homogeneous();
        Vector3 view1 = point1.homogeneous();

        // convert to camera coordinates
        Matrix3 K_inv = camera_ptr->GetIntrinsicMatrixEigen().inverse();
        view0 = K_inv * view0;
        view1 = K_inv * view1;

        // check baseline
        Vector3 baseline = pose0.translation() - pose1.translation();
        // check view angle
        // convert view0, view1 to global view direction
        view0 = pose0 * view0;
        view1 = pose1 * view1;
        view0.normalize();
        view1.normalize();

        double cos = view0.dot(view1)/(view0.norm() * view1.norm());
        auto angle = std::abs<double>(180.0 * std::acos(cos) / M_PI);
        bool state = true;
        if(parameters.baseline_check)
        {
            state = baseline.norm() >= parameters.baseline_thresh;
        }
        if(parameters.view_angle_check)
        {
            state = state & (angle > parameters.view_angle_thresh);
        }

        return state;
    }


}//end of SuperVIO
