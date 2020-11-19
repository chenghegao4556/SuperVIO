//
// Created by chenghe on 5/1/20.
//
#include <visualization/visualizer.h>
#include <ros/ros.h>

namespace SuperVIO::Visualization
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void Visualizer::
    SetData(const IMU::IMUStateMap& imu_state_map,
            const Vision::FeatureStateMap& feature_state_map,
            const Vision::TrackMap& track_map,
            const Vision::FrameMeasurementMap& frame_measurement_map,
            const Vision::Image& image)
    {
        {
            WriteLock write_lock(mutex_);
            for(const auto& imu_state: imu_state_map)
            {
                imu_state_map_.insert(imu_state);
            }
            feature_state_map_ = feature_state_map;
        }

        this->SetTrackedImage(track_map, frame_measurement_map, image);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void Visualizer::
    SetTrackedImage(const Vision::TrackMap& track_map,
                    const Vision::FrameMeasurementMap& frame_measurement_map,
                    const Vision::Image& image)
    {
        WriteLock write_lock(mutex_);
        if(!image.empty())
        {
            tracked_image_ = VisualizeTrackedImage(track_map, frame_measurement_map, image);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void Visualizer::
    ResetData()
    {
        WriteLock write_lock(mutex_);
        imu_state_map_.clear();
        feature_state_map_.clear();
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void Visualizer::
    SetFPS(const double& fps)
    {
        WriteLock write_lock(mutex_);
        if(1.0/fps < 30.0)
        {
            fps_ = fps_ * size_ + 1.0/fps;
            size_ += 1.0;
            fps_ /= size_;
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void Visualizer::
    GetData(IMU::IMUStateMap& imu_state_map,
            Vector3s& point_cloud,
            Vision::Image& tracked_image)
    {
        ReadLock read_lock(mutex_);
        imu_state_map = imu_state_map_;
        point_cloud.clear();
        for(const auto& feature: feature_state_map_)
        {
            point_cloud.push_back(feature.second.world_point);
        }
        tracked_image = tracked_image_;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void Visualizer::
    InitColorMap()
    {
        cv::Mat temp = cv::Mat::zeros(1, 255, CV_8UC1);
        for(int i = 0; i<temp.cols; i++)
        {
            temp.at<uchar>(i) = static_cast<uchar>(255 - i);
        }
        cv::applyColorMap(temp, color_map_, cv::COLORMAP_SUMMER);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Vision::Image Visualizer::
    VisualizeTrackedImage(const Vision::TrackMap& track_map,
                          const Vision::FrameMeasurementMap& frame_measurement_map,
                          const Vision::Image& image) const
    {
        auto tracked_image = image;
        if(tracked_image.channels() != 3)
        {
            cv::cvtColor(tracked_image, tracked_image, CV_GRAY2BGR);
        }

        for(const auto& track: track_map)
        {
            if(track.second.measurements.back().state_id == frame_measurement_map.rbegin()->first)
            {
                const auto& measurements = track.second.measurements;
                auto pt_0 = frame_measurement_map.at(measurements[0].state_id).key_points[measurements[0].point_id].point;
                cv::circle(tracked_image, pt_0, 1, cv::Scalar(255, 255, 0), 1, 16);
                double segment = (double)255/(double)measurements.size();
                for(size_t i = 1; i < measurements.size(); ++i)
                {
                    int color_index = static_cast<int>((double) i * segment);
                    color_index = std::min<int>(255, color_index);
                    cv::Vec3b color = color_map_.at<cv::Vec3b>(0, color_index);
                    auto pt_1 = frame_measurement_map.at(measurements[i].state_id).key_points[measurements[i].point_id].point;
                    cv::circle(tracked_image, pt_1, 1, cv::Scalar(255, 255, 0), 1, 16);
                    cv::line(tracked_image, pt_0, pt_1, color, 2, 16);
                    pt_0 = pt_1;
                }
            }
        }


        return tracked_image;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Visualizer::
    Visualizer(const Quaternion& q_i_c, Vector3 p_i_c):
        q_i_c_(q_i_c),
        p_i_c_(std::move(p_i_c)),
        fps_(0),
        size_(0)
    {
        pangolin::CreateWindowAndBind("Super VIO Viewer", 1024, 768);
        glEnable(GL_DEPTH_TEST);
        pangolin::GetBoundWindow()->RemoveCurrent();
        this->InitColorMap();
        thread_ = std::thread(std::bind(&Visualizer::RenderLoop, this));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Visualizer::Ptr Visualizer::
    Creat(const Quaternion& q_i_c, Vector3 p_i_c)
    {
        return Ptr(new Visualizer(q_i_c, p_i_c));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Visualizer::
    ~Visualizer()
    {
        thread_.join();
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void Visualizer::
    RenderLoop()
    {
        pangolin::BindToContext("Super VIO Viewer");
        glEnable(GL_DEPTH_TEST);

        pangolin::OpenGlRenderState s_cam = (pangolin::OpenGlRenderState(
                pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
                ///! camera position, model position, upper vector
                pangolin::ModelViewLookAt( 0, 0,-2, 0, 0, 0, pangolin::AxisNegY)));

        ///! for 3d interaction
        pangolin::Handler3D handler(s_cam);

        ///! display point cloud
        pangolin::View& d_cam = pangolin::CreateDisplay()
                .SetBounds(0.0, 1.0, 0.0, 1.0)
                .SetHandler(&handler);

        ///! display color image
        pangolin::View& rgb_image = pangolin::Display("color_image")
                .SetBounds(0.0, 0.3, 0.0, 0.5, 640.0/480.0)
                .SetLock(pangolin::LockLeft, pangolin::LockBottom);
        pangolin::GlTexture image_texture = pangolin::GlTexture(640, 480,
                GL_RGB, false, 0, GL_BGR,GL_UNSIGNED_BYTE);
//        //! ba log
//        pangolin::DataLog acc_bias_log;
//        acc_bias_log.SetLabels(std::vector<std::string>{"accelerator bias"});
//        pangolin::Plotter acc_bias_plotter(&acc_bias_log, 0.0f, 10.0f, 0.0f, 0.10f, 0.1f, 0.1f);
//        acc_bias_plotter.SetBounds(0.5, 0.5 + 0.5/3.0, 2.0/3.0, 1.0);
//        acc_bias_plotter.Track("$i");
//        pangolin::DisplayBase().AddDisplay(acc_bias_plotter);
//
//        //! bg log
//        pangolin::DataLog gyr_bias_log;
//        gyr_bias_log.SetLabels(std::vector<std::string>{"gyroscope bias"});
//        pangolin::Plotter gyr_bias_plotter(&gyr_bias_log, 0.0f, 10.0f, 0.0f, 0.10f, 0.1f, 0.1f);
//        gyr_bias_plotter.SetBounds(0.5 + 0.5/3.0, 0.5 + 1.0/3.0, 2.0/3.0, 1.0);
//        gyr_bias_plotter.Track("$i");
//        pangolin::DisplayBase().AddDisplay(gyr_bias_plotter);
//
//        //! velocity log
//        pangolin::DataLog velocity_log;
//        std::vector<std::string> velocity_labels{"velocity"};
//        velocity_log.SetLabels(velocity_labels);
//        pangolin::Plotter velocity_plotter(&velocity_log, 0.0f, 10.0f, 0.0f, 20.0f, 0.1f, 1.0f);
//        velocity_plotter.SetBounds(0.5 + 1.0/3.0, 1.0, 2.0/3.0, 1.0);
//        velocity_plotter.Track("$i");
//        pangolin::DisplayBase().AddDisplay(velocity_plotter);

        ///! creat buttons
        pangolin::CreatePanel("menu").SetBounds(0.8,1,0.0,0.20);
        pangolin::Var<bool> follow("menu.Follow", true, true);
        pangolin::Var<float> FPS("menu.FPS", 0.0);
        pangolin::Var<float> acc_bias("menu.ACC_BIAS", 0.0);
        pangolin::Var<float> gyr_bias("menu.GYR_BIAS", 0.0);
        pangolin::Var<float> velocity("menu.Velocity", 0.0);

        Matrix3 K = Matrix3::Identity();
        K(0, 0) = 500; K(0, 2) = 320;
        K(1, 1) = 500; K(1, 2) = 240;
        const Matrix3 K_inv = K.inverse();

        while(!pangolin::ShouldQuit() && ros::ok())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));

            Vector3s point_cloud;
            IMU::IMUStateMap imu_state_map;
            Vision::Image tracked_image;
            this->GetData(imu_state_map, point_cloud, tracked_image);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            d_cam.Activate(s_cam);
            if(imu_state_map.size() > 2)
            {
                //! draw coordinate
                glColor3f(1.0f, 0.0f, 0.0f);
                pangolin::glDrawAxis(0.2);

                //! draw cameras
                const auto& last_state = imu_state_map.rbegin()->second;
                Quaternion q_w_c = last_state.rotation * q_i_c_;
                Vector3    p_w_c = last_state.rotation * p_i_c_ + last_state.position;
                Eigen::Matrix4d T_w_c = Eigen::Matrix4d::Identity();
                T_w_c.block<3, 3>(0, 0) = q_w_c.toRotationMatrix();
                T_w_c.block<3, 1>(0, 3) = p_w_c;
                glColor3f(0.0f,0.0f,1.0f);
                pangolin::glDrawFrustum(K_inv, 640, 480, T_w_c, 0.1);

                //! draw odometry
                glLineWidth(2);
                glBegin(GL_LINES);
                glColor3f(0.0f,1.0f,0.0f);
                for(auto iter_i = imu_state_map.begin(); std::next(iter_i) != imu_state_map.end(); ++iter_i)
                {
                    auto iter_j = std::next(iter_i);
                    glVertex3f(iter_i->second.position(0), iter_i->second.position(1), iter_i->second.position(2));
                    glVertex3f(iter_j->second.position(0), iter_j->second.position(1), iter_j->second.position(2));
                }
                glEnd();

                //! draw point cloud
                glPointSize(2);
                glBegin(GL_POINTS);
                for(const auto& point: point_cloud)
                {
                    glColor3f(1.0f,1.0f,1.0f);
                    glVertex3f(point(0), point(1), point(2));
                }
                glEnd();

                //! log bias
//            gyr_bias_log.Clear();
//            acc_bias_log.Clear();
//            velocity_log.Clear();
//            float max_ba = 0.0f, max_bg = 0.0f, max_velocity = 0.0f;
//            for(const auto& imu_state: imu_state_map)
//            {
//                float ba_norm = imu_state.second.linearized_ba.norm();
//                float bg_norm = imu_state.second.linearized_bg.norm();
//                float velocity_norm = imu_state.second.velocity.norm();
//                max_ba = ba_norm > max_ba ? ba_norm : max_ba;
//                max_bg = bg_norm > max_bg ? bg_norm : max_bg;
//                max_velocity = velocity_norm > max_velocity ? velocity_norm : max_velocity;
//                acc_bias_log.Log(ba_norm);
//                gyr_bias_log.Log(bg_norm);
//                velocity_log.Log(velocity_norm);
//            }
//            acc_bias_plotter.GetView().y = pangolin::Rangef(0.0f, 1.3f * max_ba);
//            gyr_bias_plotter.GetView().y = pangolin::Rangef(0.0f, 1.3f * max_bg);
//            velocity_plotter.GetView().y = pangolin::Rangef(0.0f, 1.3f * max_velocity);
                acc_bias = imu_state_map.rbegin()->second.linearized_ba.norm();
                gyr_bias = imu_state_map.rbegin()->second.linearized_bg.norm();
                velocity = imu_state_map.rbegin()->second.velocity.norm();
                FPS = fps_;
                //! follow pose
                if(follow)
                {
                    pangolin::OpenGlMatrix mv;
                    Vector3 forward_vector(0, 0, 1);
                    Vector3 up_vector(0, -1, 0);

                    Vector3 forward = (q_w_c * forward_vector).normalized();
                    Vector3 up      = (q_w_c * up_vector).normalized();

                    Vector3 eye = p_w_c - forward;
                    Vector3 at = eye + forward;

                    Vector3 z = (eye - at).normalized();  // Forward
                    Vector3 x = up.cross(z).normalized(); // Right
                    Vector3 y = z.cross(x);

                    Eigen::Matrix4d m;
                    m << x(0),  x(1),  x(2),  -(x.dot(eye)),
                            y(0),  y(1),  y(2),  -(y.dot(eye)),
                            z(0),  z(1),  z(2),  -(z.dot(eye)),
                            0,     0,     0,              1;

                    memcpy(&mv.m[0], m.data(), sizeof(Eigen::Matrix4d));
                    s_cam.SetModelViewMatrix(mv);
                }
            }

            //! draw tracked image
            if(!tracked_image.empty())
            {
                cv::resize(tracked_image, tracked_image, cv::Size(640, 480));
                image_texture.Upload(tracked_image.data,GL_BGR,GL_UNSIGNED_BYTE);
                rgb_image.Activate();
                glColor3f(1.0,1.0,1.0);
                image_texture.RenderToViewportFlipY();
            }

            pangolin::FinishFrame();
        }
    }
}//end of SuperVIO::Visualization
