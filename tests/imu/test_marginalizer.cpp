//
// Created by chenghe on 4/25/20.
//
#include <random>
#include <fstream>
#include <imu/imu_processor.h>
#include <imu/imu_noise.h>
#include <vision/vision_measurements.h>
#include <optimization/optimizer.h>
#include <optimization/helper.h>
#include <optimization/marginalizer.h>
#include <chrono>
#include <visualization/visualizer.h>
#include <vision/camera.h>

using namespace SuperVIO;

Matrix3 Euler2Rotation( Eigen::Vector3d  eulerAngles)
{
    double roll = eulerAngles(0);
    double pitch = eulerAngles(1);
    double yaw = eulerAngles(2);

    double cr = cos(roll);  double sr = sin(roll);
    double cp = cos(pitch); double sp = sin(pitch);
    double cy = cos(yaw);   double sy = sin(yaw);

    Matrix3 RIb;
    RIb<< cy*cp,   cy*sp*sr - sy*cr,   sy*sr + cy*cr*sp,
            sy*cp,   cy*cr + sy*sr*sp,   sp*sy*cr - cy*sr,
            -sp,              cp*sr,              cp*cr;
    return RIb;
}

void CoutErrorMessage(const IMU::IMUState& true_imu_state,
                      const IMU::IMUState& optimized_state)
{
    Vector3 after_rotation_diff = Utility::EigenBase::Quaternion2Euler(optimized_state.rotation.inverse() *
                                                                       true_imu_state.rotation);
    Vector3 after_position_diff = (true_imu_state.position -  optimized_state.position);
    Vector3 after_speed_diff = (true_imu_state.velocity -  optimized_state.velocity);

    std::cout<<std::endl;
    std::cout.setf( std::ios::fixed );
    std::cout<<"*********************************"<<std::endl;
    std::cout<<"after  rotation "<<after_rotation_diff.norm()<< " after  position "<<after_position_diff.norm()
             <<" after   velocity "<<after_speed_diff.norm()<<std::endl;
}

Matrix3 EulerRates2BodyRates(Vector3 eulerAngles)
{
    double roll = eulerAngles(0);
    double pitch = eulerAngles(1);

    double cr = cos(roll);  double sr = sin(roll);
    double cp = cos(pitch); double sp = sin(pitch);

    Matrix3 R;

    R<<  1,     0,     -sp,
            0,    cr,   sr*cp,
            0,   -sr,   cr*cp;

    return R;
}

struct IMUData
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    IMUData(double _t,
            const Quaternion& _rotation, Vector3 _position,  Vector3 _velocity,
            Vector3 _acc, Vector3 _gyr,
            Vector3 _ba = Vector3::Zero(), Vector3 _bg = Vector3::Zero()):
            timestamp(_t),
            rotation(_rotation),
            position(std::move(_position)),
            velocity(std::move(_velocity)),
            acc(std::move(_acc)),
            gyr(std::move(_gyr)),
            ba(std::move(_ba)),
            bg(std::move(_bg))
    {

    }

    double timestamp;

    Quaternion rotation;
    Vector3 position;
    Vector3 velocity;

    Vector3 acc;
    Vector3 gyr;

    Vector3 ba;
    Vector3 bg;
};

IMUData CreatIMUData(double t)
{
    static const double ellipse_x = 15.0;
    static const double ellipse_y = 20.0;
    static const double z  = 1.0;
    static const double K1 = 10.0;
    static const double K  = M_PI/ 10.0;
    static const double K2 = K*K;
    static const double k_roll = 0.1;
    static const double k_pitch = 0.2;
    //gravity in navigation frame(ENU)   ENU (0,0,-9.81)  NED(0,0,9,81)
    static const Vector3 gn (0.0, 0.0, 9.81);

    // translation
    // twb:  body frame in world frame
    Vector3 position( ellipse_x * cos( K * t) + 5, ellipse_y * sin( K * t) + 5,  z * sin( K1 * K * t ) + 5);
    Vector3 dp( -K  * ellipse_x * sin(K*t),    K * ellipse_y * cos(K*t),  z*K1*K     * cos(K1 * K * t));
    Vector3 ddp(-K2 * ellipse_x * cos(K*t),  -K2 * ellipse_y * sin(K*t), -z*K1*K1*K2 * sin(K1 * K * t));

    // Rotation
    // roll ~ [-0.2, 0.2], pitch ~ [-0.3, 0.3], yaw ~ [0,2pi]
    Vector3 euler_angles(k_roll * cos(t) , k_pitch * sin(t) , K*t );
    Vector3 euler_angles_rates(-k_roll * sin(t) , k_pitch * cos(t) , K);

    // body frame to world frame
    Matrix3 rotation = Euler2Rotation(euler_angles);
    //  euler rates trans to body gyro
    Vector3 gyr = EulerRates2BodyRates(euler_angles) * euler_angles_rates;

    //  Rbw * Rwn * gn = gs
    Vector3 acc = rotation.transpose() * ( ddp +  gn );

    return IMUData(t, Quaternion(rotation), position, dp, acc, gyr);
}

class IMUNoise
{
public:
    IMUNoise():
            gyr_bias(Vector3::Zero()),
            acc_bias(Vector3::Zero())
    {

    }
    IMUData AddIMUNoise(const IMUData& data)
    {
        static const double gyro_bias_sigma = 1.0e-5;
        static const double acc_bias_sigma = 0.0001;
        static const double gyro_noise_sigma = 0.015;    // rad/s * 1/sqrt(hz)
        static const double acc_noise_sigma = 0.019;      //　m/(s^2) * 1/sqrt(hz)
        const double imu_frequency = 200.0;
        const double imu_timestep  = 1.0/imu_frequency;

        std::random_device rd;
        std::default_random_engine generator_(rd());
        std::normal_distribution<double> noise(0.0, 1.0);

        auto noise_data = data;

        Vector3 noise_gyro(noise(generator_),noise(generator_),noise(generator_));
        Matrix3 gyro_sqrt_cov = gyro_noise_sigma * Matrix3::Identity();
        noise_data.gyr = data.gyr + gyro_sqrt_cov * noise_gyro / sqrt( imu_timestep ) + gyr_bias;

        Vector3 noise_acc(noise(generator_),noise(generator_),noise(generator_));
        Matrix3 acc_sqrt_cov = acc_noise_sigma * Matrix3::Identity();
        noise_data.acc = data.acc + acc_sqrt_cov * noise_acc / sqrt( imu_timestep ) + acc_bias;

        // gyro_bias update
        Vector3 noise_gyro_bias(noise(generator_),noise(generator_),noise(generator_));
        gyr_bias += gyro_bias_sigma * sqrt(imu_timestep ) * noise_gyro_bias;
        noise_data.bg = gyr_bias;

        // acc_bias update
        Vector3 noise_acc_bias(noise(generator_),noise(generator_),noise(generator_));
        acc_bias += acc_bias_sigma * sqrt(imu_timestep ) * noise_acc_bias;
        noise_data.ba = acc_bias;

        return noise_data;
    }

    Vector3 gyr_bias;
    Vector3 acc_bias;
};

Vector3s LoadPointCloud(const std::string& file_name)
{
    std::ifstream f;
    f.open(file_name);
    Vector3s point_cloud;
    while(!f.eof())
    {
        std::string s;
        std::getline(f,s);
        if(!s.empty())
        {
            std::stringstream ss;
            ss << s;
            double x,y,z;
            ss >> x; ss >> y; ss >> z;
            point_cloud.push_back(Vector3{x, y, z});
            ss >> x; ss >> y; ss >> z;
            point_cloud.push_back(Vector3{x, y, z});
        }
    }
    std::random_device rd;
    std::default_random_engine generator_(rd());
    std::uniform_real_distribution<double> noise(-100, 100);

    int n = point_cloud.size();
    for(size_t i = 0; i < 6; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            Eigen::AngleAxisd rotation_vector (M_PI/noise(generator_),
                                               Vector3(noise(generator_), noise(generator_), noise(generator_)));
            Vector3 p = Matrix3(rotation_vector) * point_cloud[j] +
                        Vector3{noise(generator_),noise(generator_),noise(generator_)};
            point_cloud.push_back(p);
        }
    }

    return point_cloud;
}

class Timer
{
public:

    typedef std::chrono::steady_clock::time_point TimePoint;

    void Tic()
    {
        t1 = std::chrono::steady_clock::now();
    }

    void Toc()
    {
        t2 = std::chrono::steady_clock::now();
    }

    void Duration(const std::string &s)
    {
        std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>> ( t2-t1 );
        LOG(INFO) <<s<<" costs time: "<<time_used.count() <<" seconds.";
    }

private:
    TimePoint t1, t2;
};

int main()
{
    const double imu_frequency = 200.0;
    const double camera_frequency = 10.0;
    const double imu_time_step = 1.0 / imu_frequency;
    const double camera_time_step = 1.0 / camera_frequency;
    const double test_time = 20;

    const double gyro_bias_sigma = 1.0e-5 * sqrt( imu_time_step );
    const double acc_bias_sigma = 0.0001  * sqrt( imu_time_step );
    const double gyro_noise_sigma = 0.015 / sqrt( imu_time_step );    // rad/s * 1/sqrt(hz)
    const double acc_noise_sigma = 0.019 / sqrt( imu_time_step );      //　m/(s^2) * 1/sqrt(hz)

    std::vector<IMUData> imu_data_buffer;
    std::vector<IMUData> noise_data_buffer;
    double t = 0;
    //! generate imu_raw data;
    IMUNoise imu_n;
    while(t <= (test_time+imu_time_step))
    {
        imu_data_buffer.push_back(CreatIMUData(t));
        noise_data_buffer.push_back(imu_n.AddIMUNoise(imu_data_buffer.back()));
        t += imu_time_step;
    }
    const Matrix18 noise = IMU::IMUNoise::CreatNoiseMatrix(acc_noise_sigma, gyro_noise_sigma,
                                                           acc_bias_sigma,  gyro_bias_sigma);

    std::vector<double> k{500, 500, 500, 500};
    std::vector<double> d{0.0, 0.0, 0.0, 0.0};
    auto camera_ptr = Vision::Camera::Creat(
            "simplepinhole",k, d, k, cv::Size(1000, 1000), cv::Size(1000, 1000), false);


    Eigen::Matrix3d R;
    R << 0, 0, -1,
            -1, 0,  0,
            0, 1,  0;
    Quaternion q_i_c(R);
    Vector3 p_i_c{0.05,0.04,0.03};
    Visualization::Visualizer visualizer(q_i_c, p_i_c);
    const Vector3s point_cloud = LoadPointCloud(std::string("house_model/house.txt"));
    auto feature_point_cloud = point_cloud;
    std::vector<double> feature_depths;
    std::random_device rd;
    std::default_random_engine generator_(rd());
    std::normal_distribution<double> random_noise(0, 1.0);

    double current_camera_time = 0;
    double last_camera_time = 0;
    std::map<double, Optimization::BaseParametersBlock::Ptr> pose_map;
    std::map<double, Optimization::BaseParametersBlock::Ptr> speed_bias_map;
    Optimization::MarginalizationInformation marginalization_information;

    IMU::IMUStateMap ground_truth_imu_state_map;
    const Vector3 gn (0.0, 0.0, 9.81);
    IMU::IMUStateMap imu_state_map;
    IMU::IMUPreIntegrationMeasurementMap pre_map;
    IMU::IMURawMeasurementsMap raw_map;

    while(current_camera_time <= (test_time+imu_time_step) && !pangolin::ShouldQuit())
    {
        if(imu_state_map.empty())
        {
            IMU::IMUState initial_state(imu_data_buffer[0].rotation,
                          imu_data_buffer[0].position,
                          imu_data_buffer[0].velocity);
            imu_state_map.insert(std::make_pair(current_camera_time,
                                                initial_state));
            auto pose_ptr = Optimization::ParameterBlockFactory::CreatPose(
                    initial_state.rotation, initial_state.position);
            auto speed_bias_ptr = Optimization::ParameterBlockFactory::CreatSpeedBias(initial_state.velocity,
                    initial_state.linearized_ba, initial_state.linearized_bg);

            pose_map.insert(std::make_pair(current_camera_time, pose_ptr));
            speed_bias_map.insert(std::make_pair(current_camera_time, speed_bias_ptr));

            ground_truth_imu_state_map.insert(std::make_pair(current_camera_time,
                                                             IMU::IMUState(imu_data_buffer[0].rotation,
                                                                           imu_data_buffer[0].position,
                                                                           imu_data_buffer[0].velocity)));
        }
        else
        {
            IMU::IMURawMeasurements raw_measurements;
            Vector3 acc_0 = Vector3::Zero();
            Vector3 gyr_0 = Vector3::Zero();
            for(const auto& data: noise_data_buffer)
            {
                if(data.timestamp > last_camera_time && data.timestamp < current_camera_time )
                {
                    raw_measurements.emplace_back(imu_time_step, data.acc, data.gyr);
                }
                else if(data.timestamp <= last_camera_time)
                {
                    acc_0 = data.acc;
                    gyr_0 = data.gyr;
                }
            }
            const auto& last_imu_state = imu_state_map.at(last_camera_time);
            auto state_pre = IMU::IMUProcessor::Propagate(last_imu_state, acc_0, gyr_0, gn, noise, raw_measurements);
            imu_state_map.insert(std::make_pair(current_camera_time, state_pre.first));
            pre_map.insert(std::make_pair(std::make_pair(last_camera_time, current_camera_time), state_pre.second));
            raw_map.insert(std::make_pair(std::make_pair(last_camera_time, current_camera_time), raw_measurements));

            auto pose_ptr = Optimization::ParameterBlockFactory::CreatPose(
                    state_pre.first.rotation, state_pre.first.position);
            auto speed_bias_ptr = Optimization::ParameterBlockFactory::CreatSpeedBias(
                    state_pre.first.velocity, state_pre.first.linearized_ba, state_pre.first.linearized_bg);
            pose_map.insert(std::make_pair(current_camera_time, pose_ptr));
            speed_bias_map.insert(std::make_pair(current_camera_time, speed_bias_ptr));

            auto data = CreatIMUData(current_camera_time);
            ground_truth_imu_state_map.insert(std::make_pair(current_camera_time,
                                                             IMU::IMUState(data.rotation,
                                                                           data.position,
                                                                           data.velocity)));
        }
        last_camera_time = current_camera_time;
        current_camera_time += camera_time_step;
        std::map<size_t, std::map<double, cv::Point2d>> measurements;
        for(size_t i = 0; i < point_cloud.size(); ++i)
        {
            std::map<double, cv::Point2d> measurements_per_point;
            bool first = true;
            for(const auto& true_imu_state: ground_truth_imu_state_map)
            {
                const auto& time_stamp = true_imu_state.first;
                const auto& q_w_i = true_imu_state.second.rotation;
                const auto& p_w_i = true_imu_state.second.position;
                const Quaternion q_w_c = q_w_i * q_i_c;
                const Vector3    p_w_c = q_w_i * p_i_c + p_w_i;
                const Vector3 camera_point = q_w_c.inverse() * (point_cloud[i] - p_w_c);
                const Vector2 pt = camera_ptr->Project(camera_point);
                measurements_per_point.insert(std::make_pair(time_stamp,
                      cv::Point2d{pt.x() + random_noise(generator_), pt.y() + random_noise(generator_)}));
                if(first)
                {
                    feature_depths.push_back(camera_point.z());
                    first = false;
                }
            }
            measurements.insert(std::make_pair(i, measurements_per_point));
        }

        std::vector<Optimization::BaseParametersBlock::Ptr> parameter_blocks;
        std::vector<Optimization::BaseParametersBlock::Ptr> feature_parameter_blocks;

        if(pose_map.size() >= 10)
        {
            for(const auto& pose_ptr: pose_map)
            {
                parameter_blocks.push_back(pose_ptr.second);
            }
            for(const auto& speed_ptr: speed_bias_map)
            {
                parameter_blocks.push_back(speed_ptr.second);
            }
            for(const auto & i : feature_point_cloud)
            {
                const auto& time_stamp = imu_state_map.begin()->first;
                const auto& q_w_0 = imu_state_map.begin()->second.rotation;
                const auto& p_w_0 = imu_state_map.begin()->second.position;
                const Quaternion q_w_c = q_w_0 * q_i_c;
                const Vector3    p_w_c = q_w_0 * p_i_c + p_w_0;
                const Vector3 camera_point = q_w_c.inverse() * (i - p_w_c);

                auto feature_ptr = Optimization::ParameterBlockFactory::CreatInverseDepth(
                        camera_point.z());
                feature_parameter_blocks.push_back(feature_ptr);
                parameter_blocks.push_back(feature_ptr);
            }
            std::vector<Optimization::BaseResidualBlock::Ptr>   residual_blocks;
            for(size_t i = 0; i < feature_point_cloud.size(); ++i)
            {
                const auto& measurements_per_point = measurements[i];
                if(measurements_per_point.size() < 2)
                {
                    continue;
                }
                const auto& pt_0 = measurements_per_point.begin()->second;
                const auto& pose_0 = pose_map.at(measurements_per_point.begin()->first);
                const auto& feature_ptr = feature_parameter_blocks[i];
                for(auto iter = std::next(measurements_per_point.begin()); iter != measurements_per_point.end(); ++iter)
                {
                    const auto& pt_j = iter->second;
                    const auto& pose_j = pose_map.at(iter->first);

                    std::vector<Optimization::BaseParametersBlock::Ptr> pbs{pose_0, pose_j, feature_ptr};
                    auto residual_ptr = Optimization::ResidualBlockFactory::CreatReprojection(
                            camera_ptr->GetIntrinsicMatrixEigen(), q_i_c, p_i_c,
                            Vector2{pt_0.x, pt_0.y}, Vector2{pt_j.x, pt_j.y}, 1.0, 1.0, pbs);
                    residual_blocks.push_back(residual_ptr);
                }
            }
            for(const auto& pre: pre_map)
            {
                const auto& pose_index_0 = pre.first.first;
                const auto& pose_index_1 = pre.first.second;

                std::vector<Optimization::BaseParametersBlock::Ptr> pbs{pose_map.at(pose_index_0),
                                                                        speed_bias_map.at(pose_index_0),
                                                                        pose_map.at(pose_index_1),
                                                                        speed_bias_map.at(pose_index_1)};

                auto imu_residual_block = Optimization::ResidualBlockFactory::CreatPreIntegration(
                        pre.second, gn, pbs);

                residual_blocks.push_back(imu_residual_block);
            }
            if(marginalization_information.valid)
            {
                residual_blocks.push_back(Optimization::ResidualBlockFactory::CreatMarginalization(
                        marginalization_information));
            }
            Optimization::Optimizer::Options options;
            options.trust_region_strategy = "dogleg";
            options.linear_solver_type = "dense_schur";
            options.max_iteration = 10;
            options.num_threads = 4;
            //options.verbose = true;
            pose_map.begin()->second->SetFixed();
            Timer timer;
            timer.Tic();
            Optimization::Optimizer::Construct(options, parameter_blocks, residual_blocks);
            timer.Toc();
            timer.Duration("op");

            const auto origin_first_rotation = imu_state_map.begin()->second.rotation;
            const auto origin_first_position = imu_state_map.begin()->second.position;
            const auto origin_first_rotation_euler = Utility::EigenBase::Quaternion2Euler(origin_first_rotation);

            const auto optimized_first_rotation = Optimization::Helper::GetPoseFromParameterBlock(
                    pose_map.begin()->second).first;
            const auto optimized_first_position = Optimization::Helper::GetPoseFromParameterBlock(
                    pose_map.begin()->second).second;
            const auto optimized_first_rotation_euler = Utility::EigenBase::Quaternion2Euler(optimized_first_rotation);

            const auto yaw_diff = origin_first_rotation_euler.x() - optimized_first_rotation_euler.x();

            auto rotation_diff = Utility::EigenBase::EulerToQuaternion(Vector3(yaw_diff, 0, 0));
            if (abs(abs(origin_first_rotation_euler.y()) - 90)    < 1.0 ||
                abs(abs(optimized_first_rotation_euler.y()) - 90) < 1.0)
            {
                ROS_ERROR_STREAM("euler singular point!");
                rotation_diff = origin_first_rotation * optimized_first_rotation.inverse();
            }

            for(const auto& pose_ptr : pose_map)
            {
                const auto pose = Optimization::Helper::GetPoseFromParameterBlock(pose_ptr.second);
                const auto speed = Optimization::Helper::GetSpeedBiasFromParameterBlock(speed_bias_map.at(pose_ptr.first)).speed;
                const auto new_ba = Optimization::Helper::GetSpeedBiasFromParameterBlock(speed_bias_map.at(pose_ptr.first)).ba;
                const auto new_bg = Optimization::Helper::GetSpeedBiasFromParameterBlock(speed_bias_map.at(pose_ptr.first)).bg;


                const Quaternion new_rotation = rotation_diff * pose.first;
                const Vector3    new_position = rotation_diff * (pose.second - optimized_first_position) + origin_first_position;
                const Vector3    new_velocity = rotation_diff * speed;

                imu_state_map.at(pose_ptr.first).rotation = new_rotation;
                imu_state_map.at(pose_ptr.first).position = new_position;
                imu_state_map.at(pose_ptr.first).velocity = new_velocity;
                imu_state_map.at(pose_ptr.first).linearized_ba = new_ba;
                imu_state_map.at(pose_ptr.first).linearized_bg = new_bg;
                std::cout<<new_bg.transpose()<<std::endl;
            }
            for(size_t i = 0; i < feature_point_cloud.size(); ++i)
            {
                auto depth = Optimization::Helper::GetDepthFromParameterBlock(feature_parameter_blocks[i]);
                const auto& time_stamp = imu_state_map.begin()->first;
                const auto& q_w_0 = imu_state_map.begin()->second.rotation;
                const auto& p_w_0 = imu_state_map.begin()->second.position;
                const Quaternion q_w_c = q_w_0 * q_i_c;
                const Vector3    p_w_c = q_w_0 * p_i_c + p_w_0;
                const auto& measurements_per_point = measurements[i];
                const auto& pt_0 = measurements_per_point.begin()->second;
                const Vector3 camera_point = camera_ptr->BackProject(Vector2{
                    pt_0.x, pt_0.y}) * depth;
                const Vector3 world_point = q_w_c * camera_point + p_w_c;
                feature_point_cloud[i] = world_point;

            }
            for(auto& pre: pre_map)
            {
                const auto& imu_raw_measurements = raw_map.at(pre.first);
                pre.second = IMU::IMUProcessor::Repropagate(imu_state_map.at(pre.first.first).linearized_ba,
                                                            imu_state_map.at(pre.first.first).linearized_bg,
                                                            pre.second.acceleration_0, pre.second.angular_velocity_0,
                                                            noise, imu_raw_measurements);
            }
            const auto& last_t = imu_state_map.rbegin()->first;

            CoutErrorMessage(ground_truth_imu_state_map.at(last_t), imu_state_map.rbegin()->second);

            const auto& drop_imu_id =  imu_state_map.begin()->first;
            const auto& next_imu_id =  std::next(imu_state_map.begin())->first;

            std::vector<Optimization::BaseParametersBlock::Ptr> marginalized_parameter_blocks;
            std::vector<Optimization::BaseResidualBlock::Ptr>   marginalized_residual_blocks;

            std::vector<Optimization::BaseParametersBlock::Ptr> pbs{pose_map.at(drop_imu_id),
                                                                    speed_bias_map.at(drop_imu_id),
                                                                    pose_map.at(next_imu_id),
                                                                    speed_bias_map.at(next_imu_id)};

            auto imu_residual_block = Optimization::ResidualBlockFactory::CreatPreIntegration(
                    pre_map.at(std::make_pair(drop_imu_id, next_imu_id)), gn, pbs);
            marginalized_residual_blocks.push_back(imu_residual_block);

            for(size_t i = 0; i < feature_point_cloud.size(); ++i)
            {
                const auto& measurements_per_point = measurements[i];
                if(measurements_per_point.size() < 2)
                {
                    continue;
                }
                const auto& pt_0 = measurements_per_point.begin()->second;
                const auto& pose_0 = pose_map.at(measurements_per_point.begin()->first);
                const auto& feature_ptr = feature_parameter_blocks[i];
                for(auto iter = std::next(measurements_per_point.begin()); iter != measurements_per_point.end(); ++iter)
                {
                    const auto& pt_j = iter->second;
                    const auto& pose_j = pose_map.at(iter->first);

                    std::vector<Optimization::BaseParametersBlock::Ptr> pbss{pose_0, pose_j, feature_ptr};
                    auto residual_ptr = Optimization::ResidualBlockFactory::CreatReprojection(
                            camera_ptr->GetIntrinsicMatrixEigen(), q_i_c, p_i_c,
                            Vector2{pt_0.x, pt_0.y}, Vector2{pt_j.x, pt_j.y}, 1.0, 1.0, pbss);
                    marginalized_residual_blocks.push_back(residual_ptr);
                }
            }

            if(marginalization_information.valid)
            {
                marginalized_residual_blocks.push_back(Optimization::ResidualBlockFactory::CreatMarginalization(
                        marginalization_information));
            }
            const auto& drop_speed_bias_ptr = speed_bias_map.at(drop_imu_id);
            const auto& drop_pose_ptr = pose_map.at(drop_imu_id);
            std::set<Optimization::BaseParametersBlock::Ptr> parameter_blocks_set;
            for(const auto& residual_ptr: marginalized_residual_blocks)
            {
                for(const auto& parameter_ptr: residual_ptr->GetParameterBlock())
                {
                    if(parameter_blocks_set.find(parameter_ptr) == parameter_blocks_set.end())
                    {
                        parameter_blocks_set.insert(parameter_ptr);
                    }
                }
            }
            size_t i = 0;
            std::set<size_t> drop_set;
            for(const auto& parameter_ptr: parameter_blocks_set)
            {
                marginalized_parameter_blocks.push_back(parameter_ptr);
                if(parameter_ptr->GetType() == Optimization::BaseParametersBlock::Type::InverseDepth ||
                   parameter_ptr == drop_pose_ptr                          ||
                   parameter_ptr == drop_speed_bias_ptr)
                {
                    drop_set.insert(i);
                }

                ++i;
            }

            timer.Tic();
            marginalization_information = Optimization::Marginalizer::Construct(
                    marginalized_parameter_blocks, marginalized_residual_blocks, drop_set, 2);
            Vision::TrackMap track_map;
            cv::Mat image;
            Vision::FrameMeasurementMap frame_measurement_map;
            Vision::FeatureStateMap feature_state_map;
            visualizer.SetData(imu_state_map, feature_state_map, track_map, frame_measurement_map, image);
            timer.Toc();
            timer.Duration("margin");
            imu_state_map.erase(drop_imu_id);
            pose_map.erase(drop_imu_id);
            speed_bias_map.erase(drop_imu_id);
            ground_truth_imu_state_map.erase(drop_imu_id);
            raw_map.erase(std::make_pair(drop_imu_id, next_imu_id));
            pre_map.erase(std::make_pair(drop_imu_id, next_imu_id));
        }
    }

}
