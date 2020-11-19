//
// Created by chenghe on 4/20/20.
//
#include <random>
#include <fstream>
#include <imu/imu_processor.h>
#include <imu/imu_noise.h>
#include <vision/vision_measurements.h>
#include <optimization/optimizer.h>
#include <optimization/helper.h>
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
int main()
{
    const double imu_frequency = 200.0;
    const double camera_frequency = 10.0;
    const double imu_time_step = 1.0 / imu_frequency;
    const double camera_time_step = 1.0 / camera_frequency;
    const double test_time = 5;

    const double gyro_bias_sigma = 1.0e-5;
    const double acc_bias_sigma = 0.0001;
    const double gyro_noise_sigma = 0.015;    // rad/s * 1/sqrt(hz)
    const double acc_noise_sigma = 0.019;      //　m/(s^2) * 1/sqrt(hz)

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
    const Vector3 gn (0.0, 0.0, 9.81);
    IMU::IMUStateMap imu_state_map;
    IMU::IMUStateMap ground_truth_imu_state_map;
    IMU::IMUPreIntegrationMeasurementMap pre_map;
    double current_camera_time = 0;
    double last_camera_time = 0;
    DoubleVector3Map ba_map;
    DoubleVector3Map bg_map;
    while(current_camera_time <= (test_time+imu_time_step))
    {
        if(imu_state_map.empty())
        {
            imu_state_map.insert(std::make_pair(current_camera_time,
                                                IMU::IMUState(imu_data_buffer[0].rotation,
                                                              imu_data_buffer[0].position,
                                                              imu_data_buffer[0].velocity)));
            ground_truth_imu_state_map.insert(std::make_pair(current_camera_time,
                                                IMU::IMUState(imu_data_buffer[0].rotation,
                                                              imu_data_buffer[0].position,
                                                              imu_data_buffer[0].velocity)));
            ba_map.insert(std::make_pair(current_camera_time, Vector3::Zero()));
            bg_map.insert(std::make_pair(current_camera_time, Vector3::Zero()));
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
                    ba_map.insert(std::make_pair(current_camera_time, data.ba));
                    bg_map.insert(std::make_pair(current_camera_time, data.bg));
                }
            }
            const auto& last_imu_state = imu_state_map.at(last_camera_time);
            auto state_pre = IMU::IMUProcessor::Propagate(last_imu_state, acc_0, gyr_0, gn, noise, raw_measurements);
            imu_state_map.insert(std::make_pair(current_camera_time, state_pre.first));
            pre_map.insert(std::make_pair(std::make_pair(last_camera_time, current_camera_time), state_pre.second));

            auto data = CreatIMUData(current_camera_time);
            ground_truth_imu_state_map.insert(std::make_pair(current_camera_time,
                                                             IMU::IMUState(data.rotation,
                                                                           data.position,
                                                                           data.velocity)));
        }
        last_camera_time = current_camera_time;
        current_camera_time += camera_time_step;
    }

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
    const Vector3s point_cloud = LoadPointCloud(std::string("house_model/house.txt"));
    std::map<size_t, std::map<double, cv::Point2d>> measurements;
    std::vector<double> feature_depths;
    std::random_device rd;
    std::default_random_engine generator_(rd());
    std::uniform_real_distribution<double> random_noise(-2.0, 2.0);
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

    std::map<double, Optimization::BaseParametersBlock::Ptr> pose_map;
    std::map<double, Optimization::BaseParametersBlock::Ptr> speed_bias_map;
    std::vector<Optimization::BaseParametersBlock::Ptr> parameter_blocks;
    std::vector<Optimization::BaseParametersBlock::Ptr> feature_parameter_blocks;
    std::cout<<"creat pose"<<std::endl;
    for(const auto& imu_state:  imu_state_map)
    {
        auto pose_ptr = Optimization::ParameterBlockFactory::CreatPose(
                imu_state.second.rotation, imu_state.second.position);
        auto speed_bias_ptr = Optimization::ParameterBlockFactory::CreatSpeedBias(imu_state.second.velocity,
                imu_state.second.linearized_ba, imu_state.second.linearized_bg);
        pose_map.insert(std::make_pair(imu_state.first, pose_ptr));
        speed_bias_map.insert(std::make_pair(imu_state.first, speed_bias_ptr));
        parameter_blocks.push_back(pose_ptr);
        parameter_blocks.push_back(speed_bias_ptr);
    }
    pose_map.begin()->second->SetFixed();
    //speed_bias_map.begin()->second->SetFixed();

    std::cout<<"creat feature"<<std::endl;
    for(const auto& depth: feature_depths)
    {
        auto feature_ptr = Optimization::ParameterBlockFactory::CreatInverseDepth(
                depth);
        feature_parameter_blocks.push_back(feature_ptr);
        parameter_blocks.push_back(feature_ptr);
    }

    std::cout<<"creat reprojection residual"<<std::endl;
    std::vector<Optimization::BaseResidualBlock::Ptr>   residual_blocks;
    for(size_t i = 0; i < point_cloud.size(); ++i)
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

    std::cout<<"creat imu residual"<<std::endl;
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

    Optimization::Optimizer::Options options;
    options.trust_region_strategy = "dogleg";
    options.linear_solver_type = "dense_schur";
    options.max_iteration = 100;
    options.num_threads = 4;
    //options.verbose = true;
    Optimization::Optimizer::Construct(options, parameter_blocks, residual_blocks);

    const auto origin_first_rotation = imu_state_map.begin()->second.rotation;
    const auto origin_first_position = imu_state_map.begin()->second.position;
    const auto origin_first_rotation_euler = Utility::EigenBase::Quaternion2Euler(origin_first_rotation);

    const auto optimized_first_rotation = Optimization::Helper::GetPoseFromParameterBlock(
            pose_map.begin()->second).first;
    const auto optimized_first_position = Optimization::Helper::GetPoseFromParameterBlock(
            pose_map.begin()->second).second;
    const auto optimized_first_rotation_euler = Utility::EigenBase::Quaternion2Euler(optimized_first_rotation);

    const auto yaw_diff = origin_first_rotation_euler.x() - optimized_first_rotation_euler.x();

    const auto rotation_diff = Utility::EigenBase::EulerToQuaternion(Vector3(yaw_diff, 0, 0));

    for(const auto& pose_ptr : pose_map)
    {
        const auto& true_imu_state = ground_truth_imu_state_map.at(pose_ptr.first);
        const auto& noise_imu_state = imu_state_map.at(pose_ptr.first);

        const Vector3 before_rotation_diff = Utility::EigenBase::Quaternion2Euler(noise_imu_state.rotation.inverse() *
                         true_imu_state.rotation);
        const Vector3 before_position_diff = (noise_imu_state.position - true_imu_state.position);
        const Vector3 before_speed_diff    = (noise_imu_state.velocity - true_imu_state.velocity);

        const auto pose = Optimization::Helper::GetPoseFromParameterBlock(pose_ptr.second);
        const auto speed = Optimization::Helper::GetSpeedBiasFromParameterBlock(speed_bias_map.at(pose_ptr.first)).speed;
        const auto new_ba = Optimization::Helper::GetSpeedBiasFromParameterBlock(speed_bias_map.at(pose_ptr.first)).ba;
        const auto new_bg = Optimization::Helper::GetSpeedBiasFromParameterBlock(speed_bias_map.at(pose_ptr.first)).bg;

        const Vector3 old_ba = ba_map.at(pose_ptr.first);
        const Vector3 old_bg = bg_map.at(pose_ptr.first);
        const Vector3 ba_diff = new_ba - old_ba;
        const Vector3 bg_diff = new_bg - old_bg;

        const Quaternion new_rotation = rotation_diff * pose.first;
        const Vector3    new_position = rotation_diff * (pose.second - optimized_first_position) + origin_first_position;
        const Vector3    new_velocity = rotation_diff * speed;

        Vector3 after_rotation_diff = Utility::EigenBase::Quaternion2Euler(new_rotation.inverse() *
                   true_imu_state.rotation);
        Vector3 after_position_diff = (new_position -  true_imu_state.position);
        Vector3 after_speed_diff = (new_velocity -  true_imu_state.velocity);



        std::cout<<std::endl;
        std::cout.setf( std::ios::fixed );
        std::cout<<"*********************************"<<std::endl;
        std::cout<<"old ba: "<<old_ba.x()<<" "<<old_ba.y()<<" "<<old_ba.z()<<std::endl;
        std::cout<<"new ba: "<<new_ba.x()<<" "<<new_ba.y()<<" "<<new_ba.z()<<std::endl;
        std::cout<<"old bg: "<<old_bg.x()<<" "<<old_bg.y()<<" "<<old_bg.z()<<std::endl;
        std::cout<<"new bg: "<<new_bg.x()<<" "<<new_bg.y()<<" "<<new_bg.z()<<std::endl;
        std::cout<<"ba diff norm: "<<ba_diff.norm()<<" :"<<ba_diff.x()<<" "<<ba_diff.y()<<" "<<ba_diff.z()<<std::endl;
        std::cout<<"bg diff norm: "<<bg_diff.norm()<<" :"<<bg_diff.x()<<" "<<bg_diff.y()<<" "<<bg_diff.z()<<std::endl;
        std::cout<<"*********************************"<<std::endl;
        std::cout<<"before: rotation diff/degree: "<<before_rotation_diff.x()<<" "<<
                before_rotation_diff.y()<<" "<<before_rotation_diff.z()<<std::endl;
        std::cout<<"before: position diff/meter: "<<before_position_diff.x()<<" "<<
                before_position_diff.y()<<" "<<before_position_diff.z()<<std::endl;
        std::cout<<"before: velocity diff/meter: "<<before_speed_diff.x()<<" "<<
                before_speed_diff.y()<<" "<<before_speed_diff.z()<<std::endl;

        std::cout<<"*********************************"<<std::endl;
        std::cout<<"after: rotation diff/degree: "<<after_rotation_diff.x()<<" "<<
                after_rotation_diff.y()<<" "<<after_rotation_diff.z()<<std::endl;
        std::cout<<"after: position diff/meter: "<<after_position_diff.x()<<" "<<
                after_position_diff.y()<<" "<<after_position_diff.z()<<std::endl;
        std::cout<<"after: velocity diff/meter: "<<after_speed_diff.x()<<" "<<
                after_speed_diff.y()<<" "<<after_speed_diff.z()<<std::endl;
        std::cout<<"*********************************"<<std::endl;
        std::cout<<"before rotation "<<before_rotation_diff.norm()<<" before position "<<before_position_diff.norm()
                <<" before  velocity "<<before_speed_diff.norm()<<std::endl;
        std::cout<<"after  rotation "<<after_rotation_diff.norm()<< " after  position "<<after_position_diff.norm()
                <<" after   velocity "<<after_speed_diff.norm()<<std::endl;
    }

    std::cout<<point_cloud.size()<<std::endl;



    return 0;
}
