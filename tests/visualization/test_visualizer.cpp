//
// Created by chenghe on 5/1/20.
//
#include <visualization/visualizer.h>
#include <thread>

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

Vision::FeatureStateMap LoadPointCloud(const std::string& file_name)
{
    std::ifstream f;
    f.open(file_name);
    Vision::FeatureStateMap feature_state_map;
    size_t i = 0;
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
            feature_state_map.insert(std::make_pair(i, Vision::FeatureState(0, Vector3{x, y, z})));
            ++i;
            ss >> x; ss >> y; ss >> z;
            feature_state_map.insert(std::make_pair(i, Vision::FeatureState(0, Vector3{x, y, z})));
            ++i;
        }
    }

    return feature_state_map;
}

int main()
{
    std::vector<IMUData> imu_data_buffer;
    IMU::IMUStateMap imu_state_map;
    Eigen::Matrix3d R;
    R << 0, 0, -1,
            -1, 0,  0,
            0, 1,  0;
    Quaternion q_i_c(R);
    Vector3 p_i_c{0.05,0.04,0.03};

    const auto feature_state_map = LoadPointCloud(std::string("house_model/house.txt"));
    Vision::TrackMap track_map;
    cv::Mat image;
    Vision::FrameMeasurementMap frame_measurement_map;
    Visualization::Visualizer visualizer(q_i_c, p_i_c);
    double t = 0;
    while(t <= 20.0)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        auto data = CreatIMUData(t);
        imu_state_map.insert(std::make_pair(t, IMU::IMUState(data.rotation, data.position, data.velocity)));
        visualizer.SetData(imu_state_map, feature_state_map, track_map, frame_measurement_map, image);
        t += 0.1;
    }
}
