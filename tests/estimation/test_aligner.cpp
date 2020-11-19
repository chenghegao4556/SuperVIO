#include <imu/imu_processor.h>
#include <imu/imu_noise.h>
#include <imu/imu_checker.h>
#include <estimation/visual_imu_alignment.h>
#include <random>
#include <utility>
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
    static const double k_pitch = 0.20;
    //gravity in navigation frame(ENU)   ENU (0,0,-9.81)  NED(0,0,9,81)
    static const Vector3 gn (0.0, 0.0, 9.81);

    const Eigen::AngleAxisd r_w_0 (M_PI/5, Vector3(0.5,0.2,0.3));
    const Quaternion q_w_0(r_w_0);

    const Vector3 gw = q_w_0 * gn;

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
    Matrix3 rotation = q_w_0.toRotationMatrix() * Euler2Rotation(euler_angles);
    //  euler rates trans to body gyro
    Vector3 gyr = EulerRates2BodyRates(euler_angles) * euler_angles_rates;

    //  Rbw * Rwn * gn = gs
    Vector3 acc = rotation.transpose() * ( ddp +  gw );

    return IMUData(t, Quaternion(rotation), q_w_0 * position, dp, acc, gyr);
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

class InitialAlignmentFactor: public ceres::SizedCostFunction<3, 4, 1>
{
public:
    typedef IMU::PreIntegrator::StateJacobianOrder StateOrder;
    InitialAlignmentFactor(const Quaternion& q_c0_b1,
                           const Quaternion& q_c0_b2,
                           const Quaternion& q_c0_b3,
                           Vector3 p_c0_c1,
                           Vector3 p_c0_c2,
                           Vector3 p_c0_c3,
                           double gravity_norm,
                           Vector3 p_i_c,
                           const IMU::IMUPreIntegrationMeasurement& pre12,
                           const IMU::IMUPreIntegrationMeasurement& pre23):
        delta_t_12_(pre12.sum_dt),
        delta_t_23_(pre23.sum_dt),
        gravity_norm_(gravity_norm),
        p_i_c_(std::move(p_i_c)),
        delta_p_12_(pre12.delta_p),
        delta_p_23_(pre23.delta_p),
        delta_v_12_(pre12.delta_v),
        ba_(pre12.linearized_ba),
        r1_(q_c0_b1.toRotationMatrix()),
        r2_(q_c0_b2.toRotationMatrix()),
        r3_(q_c0_b3.toRotationMatrix()),
        p1_(std::move(p_c0_c1)),
        p2_(std::move(p_c0_c2)),
        p3_(std::move(p_c0_c3))
    {
        sqrt_information_ = Eigen::LLT<Matrix3>(pre23.covariance.block<3, 3>(0, 0).inverse()).matrixL().transpose();
        //sqrt_information_ = Matrix3::Identity();
    }


    bool Evaluate(double const *const *parameters,
                  double *residuals,
                  double **jacobians) const override
    {
        ConstMapQuaternion q_c0_w(parameters[0]);
        const double scale = parameters[1][0];

        const Vector3 gw{0, 0, gravity_norm_};

        const Vector3 v1 = (scale * r1_.transpose() * (p2_ - p1_) -
                r1_.transpose() * r2_ * p_i_c_ + p_i_c_ +
                0.5 * delta_t_12_ * delta_t_12_ * r1_.transpose() * (q_c0_w * gw) -
                delta_p_12_) / delta_t_12_;

        const Vector3 v2 = (r2_.transpose() * r1_) * (delta_v_12_ + v1)
                -delta_t_12_ * (r2_.transpose() * q_c0_w * gw);

        Eigen::Map<Eigen::Matrix<double, 3, 1>> residual_vector(residuals);

        residual_vector = scale * r2_.transpose() * (p3_ - p2_)  -
                r2_.transpose() * r3_ * p_i_c_ + p_i_c_ +
                0.5 * delta_t_23_ * delta_t_23_ * r2_.transpose() * q_c0_w * gw -
                v2 * delta_t_23_ - delta_p_23_;


        residual_vector = sqrt_information_ * residual_vector;

        if (jacobians)
        {
            Matrix3 skew_gw =  Utility::EigenBase::SkewSymmetric(gw);

            Matrix3 dv1_dqw = -0.5 * delta_t_12_ * r1_.transpose() * q_c0_w.toRotationMatrix() * skew_gw;
            Vector3 dv1_ds  =  r1_.transpose() * (p2_ - p1_) / delta_t_12_;

            Matrix3 dv2_dv1 = r2_.transpose() * r1_;
            Matrix3 dv2_dqw = delta_t_12_ * r2_.transpose() * q_c0_w * skew_gw;

            Matrix3 dr_dv2 = -delta_t_23_ * Matrix3::Identity();
            Matrix3 dr_dqw = -0.5 * delta_t_23_ * delta_t_23_ * r2_.transpose() * q_c0_w.toRotationMatrix() * skew_gw;
            Vector3 dr_ds  =  r2_.transpose() * (p3_ - p2_);


            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jacobian_qw(jacobians[0]);
                jacobian_qw.setZero();
                jacobian_qw.block<3, 3>(0, 0) = (dr_dqw + dr_dv2 * (dv2_dqw + dv2_dv1 * dv1_dqw)).block<3, 3>(0, 0);

                jacobian_qw = sqrt_information_ * jacobian_qw;

            }

            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 3, 1>> jacobian_bs(jacobians[1]);
                jacobian_bs.setZero();
                jacobian_bs = dr_ds + dr_dv2 * dv2_dv1 * dv1_ds;

                jacobian_bs = sqrt_information_ * jacobian_bs;
            }
        }

        return true;

    }

    const double delta_t_12_;
    const double delta_t_23_;

    const double gravity_norm_;

    const Vector3 p_i_c_;

    const Vector3 delta_p_12_;
    const Vector3 delta_p_23_;

    const Vector3 delta_v_12_;

    const Vector3 ba_;

    const Matrix3 r1_;
    const Matrix3 r2_;
    const Matrix3 r3_;

    const Vector3 p1_;
    const Vector3 p2_;
    const Vector3 p3_;

    Matrix3 sqrt_information_;

};//end of InitialAlignmentFactor

class QuaternionParameterization : public ceres::LocalParameterization
{
public:

    ~QuaternionParameterization() override = default;

    bool Plus(const double* x,
              const double* delta_x,
              double* x_plus_delta_x) const override
    {
        ConstMapQuaternion q(x);
        const Quaternion dq = Utility::EigenBase::DeltaQ(Vector3{delta_x[0], delta_x[1], 0});
        Eigen::Map<Eigen::Quaterniond> new_q(x_plus_delta_x);
        new_q = (q * dq).normalized();

        return true;
    }

    bool ComputeJacobian(const double* x, double* jacobian) const override
    {
        Eigen::Map<Eigen::Matrix<double, 4, 2, Eigen::RowMajor>> j(jacobian);
        j.topRows<2>().setIdentity();
        j.bottomRows<2>().setZero();

        return true;
    }


    [[nodiscard]] int GlobalSize() const override
    {
        return 4;
    }


    [[nodiscard]] int LocalSize() const override
    {
        return 2;
    }

};// PoseLocalParameterization

int main()
{
    const double imu_frequency = 200.0;
    const double camera_frequency = 10.0;
    const double imu_time_step = 1.0 / imu_frequency;
    const double camera_time_step = 1.0 / camera_frequency;
    const double test_time = 1;

    const double gyro_bias_sigma = 1.0e-5 * sqrt( imu_time_step );
    const double acc_bias_sigma = 0.0001  * sqrt( imu_time_step );
    const double gyro_noise_sigma = 0.015 / sqrt( imu_time_step );    // rad/s * 1/sqrt(hz)
    const double acc_noise_sigma = 0.019 / sqrt( imu_time_step );      //　m/(s^2) * 1/sqrt(hz)

    const Matrix18 noise = IMU::IMUNoise::CreatNoiseMatrix(acc_noise_sigma, gyro_noise_sigma,
                                                           acc_bias_sigma,  gyro_bias_sigma);
    const Vector3 gn (0.0, 0.0, 9.81);

    std::vector<IMUData> imu_data_buffer;
    std::vector<IMUData> noise_data_buffer;
    double t = 0;
    //! generate imu_raw data;
    IMUNoise imu_n;
    imu_n.acc_bias = Vector3{0.5, 0.6, 0.7};
    imu_n.gyr_bias = Vector3{0.5, 0.6, 0.2};
    while(t <= (test_time+imu_time_step))
    {
        imu_data_buffer.push_back(CreatIMUData(t));
        noise_data_buffer.push_back(imu_n.AddIMUNoise(imu_data_buffer.back()));
        t += imu_time_step;
    }

    Eigen::Matrix3d R;
    R << 0, 0, -1,
        -1, 0,  0,
         0, 1,  0;
    Quaternion q_i_c(R);
    Vector3 p_i_c{0.05,0.04,0.03};

    std::map<Vision::StateKey, SFM::Pose> visual_pose_map;
    Estimation::VIOStatesMeasurements states_measurements;

    const Quaternion q_w_b0 = imu_data_buffer[0].rotation;
    const Vector3    p_w_b0 = imu_data_buffer[0].position;

    double current_camera_time = 0;
    double last_camera_time = 0;
    while(current_camera_time <= (test_time+imu_time_step))
    {
        if(states_measurements.imu_state_map.empty())
        {
            states_measurements.imu_state_map.insert(std::make_pair(current_camera_time,
                                                IMU::IMUState(imu_data_buffer[0].rotation,
                                                              imu_data_buffer[0].position,
                                                              imu_data_buffer[0].velocity)));
            visual_pose_map.insert(std::make_pair(current_camera_time,
                    SFM::Pose(true, imu_data_buffer[0].rotation,
                              (imu_data_buffer[0].position + imu_data_buffer[0].rotation * p_i_c)/1.0)));
        }
        else
        {
            IMU::IMURawMeasurements raw_measurements;
            Vector3 acc_0 = Vector3::Zero();
            Vector3 gyr_0 = Vector3::Zero();
            for(const auto& data: noise_data_buffer)
            {
                if(data.timestamp > last_camera_time && data.timestamp <= current_camera_time )
                {
                    raw_measurements.emplace_back(imu_time_step, data.acc, data.gyr);
                }
                else if(data.timestamp <= last_camera_time)
                {
                    acc_0 = data.acc;
                    gyr_0 = data.gyr;
                }
            }
            const auto& last_imu_state = states_measurements.imu_state_map.at(last_camera_time);
            auto state_pre = IMU::IMUProcessor::Propagate(last_imu_state, acc_0, gyr_0, gn, noise, raw_measurements);
            states_measurements.imu_state_map.insert(std::make_pair(current_camera_time, state_pre.first));
            states_measurements.imu_pre_integration_measurements_map.insert(std::make_pair(
                    std::make_pair(last_camera_time, current_camera_time), state_pre.second));
            states_measurements.imu_raw_measurements_map.insert(std::make_pair(
                    std::make_pair(last_camera_time, current_camera_time), raw_measurements));

            auto data = CreatIMUData(current_camera_time);
            visual_pose_map.insert(std::make_pair(current_camera_time,
                                                  SFM::Pose(true, data.rotation,
                                                            (data.rotation * p_i_c + data.position)/1.0)));
        }
        last_camera_time = current_camera_time;
        current_camera_time += camera_time_step;
    }
    bool status = IMU::IMUCheck::IsFullyExcited(states_measurements.imu_pre_integration_measurements_map);

    if(status)
    {
        std::cout<<"full ex"<<std::endl;
    }

    Vector3 delta_bg = Estimation::VisualIMUAligner::SolveGyroscopeBias(states_measurements, visual_pose_map);
    std::cout<<"estimated bg: "<<delta_bg.transpose()<<std::endl;
    for(auto& imu_state: states_measurements.imu_state_map)
    {
        imu_state.second.linearized_bg += delta_bg;
    }
    for(auto& pre: states_measurements.imu_pre_integration_measurements_map)
    {
        auto imu_state = states_measurements.imu_state_map.at(pre.first.first);
        const auto& imu_raw_measurements = states_measurements.imu_raw_measurements_map.at(pre.first);
        pre.second = IMU::IMUProcessor::Repropagate(imu_state.linearized_ba,
                                                    imu_state.linearized_bg,
                                                    pre.second.acceleration_0, pre.second.angular_velocity_0,
                                                    noise, imu_raw_measurements);
    }
    auto result = Estimation::VisualIMUAligner::SolveGravityVectorAndVelocities(
            states_measurements, visual_pose_map, p_i_c, 9.81);
    std::cout<<result.gravity_vector.transpose()<<std::endl;
    std::cout<<"scale "<<result.scale<<std::endl;

    const Eigen::AngleAxisd r_w_0 (M_PI/5, Vector3(0.5,0.2,0.3));
    const Quaternion q_w_0(r_w_0);

    const Vector3 gw = q_w_0 * gn;

    std::cout<<"true gw:"<<gw.transpose()<<std::endl;
    double ba[3];
    ba[0] = 0; ba[1] = 0; ba[2] = 0;
    double scale = result.scale;
    auto* q_i0_w = new double[4]();

    const Vector3 Gn = result.gravity_vector.normalized();
    const Vector3 Gw = Vector3{0, 0, 9.81}.normalized();
    Vector3 axis = Gw.cross(Gn).normalized();
    double theta = std::atan2(Gw.cross(Gn).norm(), Gw.dot(Gn));

    Quaternion q_w{Eigen::AngleAxisd{theta, axis}};
    q_i0_w[0] = q_w.x(); q_i0_w[1] = q_w.y(); q_i0_w[2] = q_w.z(); q_i0_w[3] = q_w.w();
    ceres::Problem problem;
    ceres::LocalParameterization* parameterization = new QuaternionParameterization();
    problem.AddParameterBlock(q_i0_w, 4, parameterization);

    auto iter_1 = visual_pose_map.begin();
    for(; std::next(std::next(iter_1)) != visual_pose_map.end(); ++iter_1)
    {
        auto iter_2 = std::next(iter_1);
        auto iter_3 = std::next(iter_2);
        const Quaternion q_c0_b1 = visual_pose_map.at(iter_1->first).rotation;
        const Vector3    p_c0_c1 = visual_pose_map.at(iter_1->first).position;
        const Quaternion q_c0_b2 = visual_pose_map.at(iter_2->first).rotation;
        const Vector3    p_c0_c2 = visual_pose_map.at(iter_2->first).position;
        const Quaternion q_c0_b3 = visual_pose_map.at(iter_3->first).rotation;
        const Vector3    p_c0_c3 = visual_pose_map.at(iter_3->first).position;

        const auto& pre12 = states_measurements.imu_pre_integration_measurements_map.at(std::make_pair(
                iter_1->first, iter_2->first));
        const auto& pre23 = states_measurements.imu_pre_integration_measurements_map.at(std::make_pair(
                iter_2->first, iter_3->first));
        auto cost_function = new InitialAlignmentFactor(
                q_c0_b1, q_c0_b2, q_c0_b3, p_c0_c1, p_c0_c2, p_c0_c3, 9.81, p_i_c, pre12, pre23);
        problem.AddResidualBlock(cost_function, nullptr, q_i0_w, &scale);
    }

    std::cout<<"start"<<std::endl;
    Optimization::Optimizer::Options options;
    options.max_iteration = 100;
    options.num_threads = 1;
    ceres::Solver::Summary summary;
    ceres::Solve(options.ToCeres(), &problem , &summary);

    //std::cout << summary.FullReport() << "\n";

    q_w = Quaternion(q_i0_w[3], q_i0_w[0], q_i0_w[1], q_i0_w[2]);

    std::cout<<"scale: "<<scale<<std::endl;
    std::cout<<"ba: "<<Vector3(ba).transpose()<<std::endl;
    std::cout<<"op: "<<(q_w * Vector3{0, 0, 9.81}).transpose()<<std::endl;

    Matrix3 r_v_i;
    r_v_i<<9.999976e-01, 7.553071e-04, -2.035826e-03,
           -7.854027e-04, 9.998898e-01, -1.482298e-02,
            2.024406e-03, 1.482454e-02, 9.998881e-01;
    Vector3 t_v_i;
    t_v_i <<-8.086759e-01, 3.195559e-01, -7.997231e-01;
    Matrix3 r_c_v;
    r_c_v <<7.533745e-03, -9.999714e-01, -6.166020e-04,
            1.480249e-02, 7.280733e-04, -9.998902e-01,
            9.998621e-01, 7.523790e-03, 1.480755e-02;
    Vector3 t_c_v;
    t_c_v <<-4.069766e-03, -7.631618e-02, -2.717806e-01;

    Matrix3 r_c_i = r_c_v * r_v_i;
    Vector3 t_c_i = r_c_v * t_v_i + t_c_v;
    Matrix3 r_i_c = r_c_i.transpose();
    Vector3 t_i_c = -r_c_i.transpose() * t_c_i;
    std::cout<<"r_i_c:"<<std::endl;
    std::cout<<r_i_c<<std::endl;
    std::cout<<"t_i_c:"<<std::endl;
    std::cout<<t_i_c<<std::endl;
    std::cout<<"r_c_i:"<<std::endl;
    std::cout<<r_c_i<<std::endl;
    std::cout<<"t_c_i:"<<std::endl;
    std::cout<<t_c_i<<std::endl;

}
