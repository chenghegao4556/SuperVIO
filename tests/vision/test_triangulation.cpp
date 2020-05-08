//
// Created by chenghe on 3/12/20.
//
#include <gtest/gtest.h>
#include <vision/triangulator.h>
TEST(TriangulationTest, BaseTest)
{
    using namespace SuperVIO;

    Vision::Triangulator::Parameters p(true, true, 1, 10, 10);

    std::vector<double> k{500, 500, 500, 500};
    std::vector<double> d{0.0, 0.0, 0.0, 0.0};
    auto camera_ptr = Vision::Camera::Creat(
            "simplepinhole",k, d, k, cv::Size(1000, 1000), cv::Size(1000, 1000), false);

    ASSERT_TRUE(camera_ptr != nullptr);

    Vector3 world_point{50, 50, 50};

    std::vector<Vector3, Eigen::aligned_allocator<Vector3>> positions;
    std::vector<Quaternion, Eigen::aligned_allocator<Quaternion>> rotations;
    std::vector<cv::Point2f> pts;

    positions.emplace_back(0, 0, 0);
    rotations.push_back(Quaternion::Identity());
    auto c0 = camera_ptr->Project(world_point);
    pts.emplace_back(c0.x(), c0.y());
    pts[0] += cv::Point2f(0.6, 0.8);

    positions.emplace_back(10, 10, 0);
    Eigen::AngleAxisd rotation_vector1(M_PI/8, Eigen::Vector3d(1, 1, 0));
    rotations.emplace_back(rotation_vector1);
    auto world_point_1 = Utility::EigenBase::ToPose3(rotations[1], positions[1]).inverse() * world_point;
    auto c1 = camera_ptr->Project(world_point_1);
    pts.emplace_back(c1.x(), c1.y());
    pts[1] += cv::Point2f(-0.5, 0.9);


    auto r = Vision::Triangulator::TriangulatePoints(positions, rotations, pts, Quaternion::Identity(), Vector3::Zero(),
            camera_ptr, p);

    ASSERT_TRUE(r.status == Vision::Triangulator::Status::Success);
    ASSERT_TRUE((r.world_point - world_point).norm() < 5)<<r.world_point;
    ASSERT_TRUE(std::abs(r.depth - world_point.z())  < 2)<<r.depth;
    std::cout<<"world point: ["<<world_point.x()<<", "<<world_point.y()<<", "<<world_point.x()<<"]"<<std::endl;
    std::cout<<"result: ["<<r.world_point.x()<<", "<<r.world_point.y()<<", "<<r.world_point.x()<<"]"<<std::endl;
    std::cout<<"mean reprojection error: "<<r.mean_reprojection_error<<std::endl;
    std::cout<<"depth : "<<r.depth<<std::endl;

    positions.emplace_back(15, 10, 0);
    Eigen::AngleAxisd rotation_vector2(-M_PI/8, Eigen::Vector3d(0, 1, 1));
    rotations.emplace_back(rotation_vector2);
    auto world_point_2 = Utility::EigenBase::ToPose3(rotations[2], positions[2]).inverse() * world_point;
    auto c2 = camera_ptr->Project(world_point_2);
    pts.emplace_back(c2.x(), c2.y());
    pts[2] += cv::Point2f(0.0, -1.1);

    r = Vision::Triangulator::TriangulatePoints(positions, rotations, pts, Quaternion::Identity(), Vector3::Zero(),
                                                camera_ptr, p);

    ASSERT_TRUE(r.status == Vision::Triangulator::Status::Success);
    ASSERT_TRUE((r.world_point - world_point).norm() < 5)<<r.world_point;
    ASSERT_TRUE(std::abs(r.depth - world_point.z())  < 2)<<r.depth;
    std::cout<<"world point: ["<<world_point.x()<<", "<<world_point.y()<<", "<<world_point.x()<<"]"<<std::endl;
    std::cout<<"result: ["<<r.world_point.x()<<", "<<r.world_point.y()<<", "<<r.world_point.x()<<"]"<<std::endl;
    std::cout<<"mean reprojection error: "<<r.mean_reprojection_error<<std::endl;
    std::cout<<"depth : "<<r.depth<<std::endl;


}

