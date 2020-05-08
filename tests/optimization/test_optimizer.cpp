//
// Created by chenghe on 4/16/20.
//
#include <random>
#include <gtest/gtest.h>
#include <vision/triangulator.h>
#include <optimization/optimizer.h>
#include <optimization/helper.h>
#include <fstream>
#include "ceres/rotation.h"

template<typename T>
void FscanfOrDie(FILE* fptr, const char* format, T* value)
{
    int num_scanned = fscanf(fptr, format, value);
    if (num_scanned != 1) {
        LOG(FATAL) << "Invalid UW data file.";
    }
}
using namespace SuperVIO;
Quaternion
AngleAxisToQuaternion(const Vector3& angle_axis)
{
    const auto& a0 = angle_axis(0);
    const auto& a1 = angle_axis(1);
    const auto& a2 = angle_axis(2);
    const auto theta_squared = a0 * a0 + a1 * a1 + a2 * a2;

    double w, x, y, z;
    if (theta_squared > 0.0)
    {
        const double theta = sqrt(theta_squared);
        const double half_theta = theta * 0.5;
        const double k = sin(half_theta) / theta;

        w = cos(half_theta);
        x = a0 * k;
        y = a1 * k;
        z = a2 * k;
    } else
    {
        const double k = 0.5;
        w = 1.0;
        x = a0 * k;
        y = a1 * k;
        z = a2 * k;
    }
    return Quaternion(w, x, y, z).normalized();
}

cv::Point2d Distortion(const double& k1, const double& k2, const cv::Point2d& p_u)
{

    double mx2_u, my2_u, rho2_u, rad_dist_u;

    mx2_u = p_u.x * p_u.x;
    my2_u = p_u.y * p_u.y;
    rho2_u = mx2_u + my2_u;
    rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
    cv::Point2d p_d(p_u.x * rad_dist_u, p_u.y * rad_dist_u);

    return p_d;
}

void WriteToPLYFile(const Vector3s& points,
                    const Quaternions& rotations,
                    const Vector3s& positions,
                    const std::string& filename);
std::normal_distribution<> norm {0.0, 1.0};
std::random_device rd;
std::default_random_engine rng {rd()};
Vector3 PerturbPoint3(const Vector3& point)
{
    return point + Vector3{norm(rng), norm(rng), norm(rng)};
}
class BALProblem
{
public:
    explicit BALProblem(const std::string& filename)
    {
        FILE* fptr = fopen(filename.c_str(), "r");

        if (fptr == nullptr)
        {
            std::cout << "Error: unable to open file " << filename<<std::endl;
            return;
        };

        // This wil die horribly on invalid files. Them's the breaks.
        FscanfOrDie(fptr, "%d", &num_cameras_);
        FscanfOrDie(fptr, "%d", &num_points_);
        FscanfOrDie(fptr, "%d", &num_observations_);

        std::cout << "Header: " << num_cameras_ << " " << num_points_ << " " << num_observations_<<std::endl;

        for (int i = 0; i < num_observations_; ++i)
        {
            int pose_index, point_index;
            FscanfOrDie(fptr, "%d", &pose_index);
            FscanfOrDie(fptr, "%d", &point_index);
            cv::Point2d pt;
            FscanfOrDie(fptr, "%lf", &pt.x);
            FscanfOrDie(fptr, "%lf", &pt.y);
            measurements[point_index][pose_index] = -pt;
        }
        std::vector<std::vector<double>> intrinsics;
        for (int i = 0; i < num_cameras_; ++i)
        {
            std::vector<double> data_buffer;
            for(size_t j = 0; j < 9; ++j)
            {
                double data;
                FscanfOrDie(fptr, "%lf", &data);
                data_buffer.push_back(data);
            }
            rotations.push_back(AngleAxisToQuaternion(Vector3{data_buffer[0], data_buffer[1], data_buffer[2]}));
            positions.push_back(Vector3{data_buffer[3], data_buffer[4], data_buffer[5]});
            rotations[i] = rotations[i].inverse();
            positions[i] = -(rotations[i] * positions[i]);
            intrinsics.push_back(std::vector<double>{data_buffer[6], data_buffer[7], data_buffer[8]});
        }
        for(size_t i = 0; i < num_points_; ++i)
        {
            std::vector<double> data_buffer;
            for(size_t j = 0; j < 3; ++j)
            {
                double data;
                FscanfOrDie(fptr, "%lf", &data);
                data_buffer.push_back(data);
            }

            point_cloud.push_back(Vector3{data_buffer[0], data_buffer[1], data_buffer[2]});
        }
        std::cout<<"start undistort "<<std::endl;
        for(auto& m: measurements)
        {
            for(auto& pt2: m.second)
            {
                const double distort_u = pt2.second.x;
                const double distort_v = pt2.second.y;

                const auto& intrinsic = intrinsics[pt2.first];
                const auto& focal = intrinsic[0];
                const auto& k1    = intrinsic[1];
                const auto& k2    = intrinsic[2];

                const double distort_x = distort_u / focal;
                const double distort_y = distort_v / focal;

                int n = 8;
                auto d_u = Distortion(k1, k2, cv::Point2d(distort_x, distort_y));
                cv::Point2d p_u;
                // Approximate value
                p_u.x = distort_x - d_u.x;
                p_u.y = distort_y - d_u.y;

                for (int i = 1; i < n; ++i)
                {
                    d_u = Distortion(k1, k2, cv::Point2f(p_u.x, p_u.x));
                    p_u.x = distort_x - d_u.x;
                    p_u.y = distort_y - d_u.y;
                }
                p_u.x = 500 * p_u.x + 500;
                p_u.y = 500 * p_u.y + 500;
                pt2.second = p_u;
            }
        }
        std::cout<<"finished undistort "<<std::endl;
        for(auto & position : positions)
        {
            position = PerturbPoint3(position);
        }
        for(auto & point : point_cloud)
        {
            point = PerturbPoint3(point);
        }
        WriteToPLYFile(point_cloud, rotations, positions, std::string("./begin.ply"));
    }

    int num_cameras_;
    int num_points_;
    int num_observations_;

    Vector3s      positions;
    Quaternions   rotations;
    Vector3s    point_cloud;
    //! point_id, camera_id
    std::map<size_t, std::map<size_t, cv::Point2d>> measurements;
};

void WriteToPLYFile(const Vector3s& points,
                    const Quaternions& rotations,
                    const Vector3s& positions,
                    const std::string& filename)
{
    std::ofstream of(filename.c_str());

    of << "ply"
       << '\n' << "format ascii 1.0"
       << '\n' << "element vertex " << points.size() + rotations.size()
       << '\n' << "property float x"
       << '\n' << "property float y"
       << '\n' << "property float z"
       << '\n' << "property uchar red"
       << '\n' << "property uchar green"
       << '\n' << "property uchar blue"
       << '\n' << "end_header" << std::endl;

    for (const auto & position : positions)
    {
        of << position.x() << ' ' << position.y() << ' ' << position.z()
           << " 0 255 0" << '\n';
    }

    for (const auto & point : points)
    {
        of << point.x() << ' ' << point.y() << ' ' << point.z()
           << " 255 255 255" << '\n';
    }
    of.close();
}

int main()
{
    std::vector<double> k{500, 500, 500, 500};
    std::vector<double> d{0.0, 0.0, 0.0, 0.0};
    auto camera_ptr = Vision::Camera::Creat(
            "simplepinhole",k, d, k, cv::Size(1000, 1000), cv::Size(1000, 1000), false);


    BALProblem bal("./123.txt");
    std::vector<Optimization::BaseParametersBlock::Ptr> pose_parameter_blocks;
    std::vector<Optimization::BaseParametersBlock::Ptr> feature_parameter_blocks;
    std::vector<Optimization::BaseResidualBlock::Ptr>   residual_blocks;
    std::cout<<"creat pose"<<std::endl;
    for(size_t i = 0; i < bal.rotations.size(); ++i)
    {
        auto pose_ptr = Optimization::ParameterBlockFactory::CreatPose(
                bal.rotations[i], bal.positions[i]);
        pose_parameter_blocks.push_back(pose_ptr);
    }
    pose_parameter_blocks.front()->SetFixed();
    std::cout<<"creat feature"<<std::endl;
    for(size_t i = 0; i < bal.point_cloud.size(); ++i)
    {
        const auto& world_point = bal.point_cloud[i];
        const size_t pose_index_0 = bal.measurements[i].begin()->first;
        const auto& position_0 = bal.positions[pose_index_0];
        const auto& rotation_0 = bal.rotations[pose_index_0];

        const Vector3 camera_point = rotation_0.inverse() * (world_point - position_0);
        const double depth = camera_point.z();
        auto feature_ptr = Optimization::ParameterBlockFactory::CreatInverseDepth(
                depth);
        feature_parameter_blocks.push_back(feature_ptr);
    }
    feature_parameter_blocks.front()->SetFixed();
    std::cout<<"creat residual"<<std::endl;
    for(size_t i = 0; i < bal.point_cloud.size(); ++i)
    {
        const auto& measurements = bal.measurements[i];
        if(measurements.size() < 2)
        {
            continue;
        }
        const auto& pt_0 = measurements.begin()->second;
        const auto& pose_0 = pose_parameter_blocks[measurements.begin()->first];
        const auto& feature_ptr = feature_parameter_blocks[i];
        for(auto iter = std::next(measurements.begin()); iter != measurements.end(); ++iter)
        {
            const auto& pt_j = iter->second;
            const auto& pose_j = pose_parameter_blocks[iter->first];

            std::vector<Optimization::BaseParametersBlock::Ptr> parameter_blocks{pose_0, pose_j, feature_ptr};
            auto residual_ptr = Optimization::ResidualBlockFactory::CreatReprojection(
                    camera_ptr->GetIntrinsicMatrixEigen(), Quaternion::Identity(), Vector3::Zero(),
                    Vector2{pt_0.x, pt_0.y}, Vector2{pt_j.x, pt_j.y}, 1.0, 1.0, parameter_blocks);

            residual_blocks.push_back(residual_ptr);
        }
    }
    std::vector<Optimization::BaseParametersBlock::Ptr> parameter_blocks;
    for(const auto& pose: pose_parameter_blocks)
    {
        parameter_blocks.push_back(pose);
    }

    for(const auto& feature: feature_parameter_blocks)
    {
        parameter_blocks.push_back(feature);
    }
    std::cout<<"op"<<std::endl;
    Optimization::Optimizer::Options options;
    options.trust_region_strategy = "dogleg";
    options.linear_solver_type = "dense_schur";
    options.max_iteration = 100;
    options.num_threads = 8;
    Optimization::Optimizer::Construct(options, parameter_blocks, residual_blocks);
    Vector3s positions;
    Quaternions rotations;
    Vector3s points;
    for(const auto& pose_ptr : pose_parameter_blocks)
    {
        auto pose = Optimization::Helper::GetPoseFromParameterBlock(pose_ptr);
        positions.push_back(pose.second);
        rotations.push_back(pose.first);
    }
    for(size_t i = 0; i < feature_parameter_blocks.size(); ++i)
    {
        auto feature_ptr = feature_parameter_blocks[i];
        auto depth = (1.0/(*feature_ptr->GetData()));
        const auto& pt_0 = bal.measurements[i].begin()->second;
        Vector3 world_point_0 = camera_ptr->BackProject(Vector2{pt_0.x,
                                                                pt_0.y}) * depth;

        Vector3 world_point = rotations[bal.measurements[i].begin()->first]  * world_point_0 +
                positions[bal.measurements[i].begin()->first];
        points.push_back(world_point);
    }
    double error = 0;
    size_t in = 0;
    for(size_t i = 0; i < bal.point_cloud.size(); ++i)
    {
        const auto& world_point = bal.point_cloud[i];
        const size_t pose_index_0 = bal.measurements[i].begin()->first;
        const auto& position_0 = bal.positions[pose_index_0];
        const auto& rotation_0 = bal.rotations[pose_index_0];

        const Vector3 camera_point = rotation_0.inverse() * (world_point - position_0);
        const double depth = camera_point.z();

        const auto pt_0 = bal.measurements[i].begin()->second;
        Vector3 point = camera_ptr->BackProject(Vector2{
            pt_0.x, pt_0.y}) * depth;
        point = rotation_0 * point + position_0;
        for(const auto& m: bal.measurements[i])
        {
            const auto pose_index = m.first;
            const auto& rotation = bal.rotations[pose_index];
            const auto& position = bal.positions[pose_index];

            const Vector3 camera_point_j = rotation.inverse() * (point - position);
            const double p_x = camera_point_j.x() / camera_point_j.z() * 500 + 500;
            const double p_y = camera_point_j.y() / camera_point_j.z() * 500 + 500;

            const double error_x = p_x - m.second.x;
            const double error_y = p_y - m.second.y;
            error += std::sqrt(error_x * error_x + error_y * error_y);
            in ++;
        }
    }

    std::cout<<"before: "<<error/(double)in<<std::endl;

    double op_error = 0;
    size_t op_in = 0;
    for(size_t i = 0; i < points.size(); ++i)
    {
        const auto& point = points[i];
        for(const auto& m: bal.measurements[i])
        {
            const auto pose_index = m.first;
            const auto& rotation = rotations[pose_index];
            const auto& position = positions[pose_index];

            const Vector3 camera_point = rotation.inverse() * (point - position);
            const double p_x = camera_point.x() / camera_point.z() * 500 + 500;
            const double p_y = camera_point.y() / camera_point.z() * 500 + 500;

            const double error_x = p_x - m.second.x;
            const double error_y = p_y - m.second.y;
            op_error += std::sqrt(error_x * error_x + error_y * error_y);
            op_in ++;
        }
    }

    std::cout<<"after: "<<op_error/(double)op_in<<std::endl;

    WriteToPLYFile(points, rotations, positions, std::string("./final.ply"));
}
