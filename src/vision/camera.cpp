//
// Created by chenghe on 5/3/20.
//
#include <vision/camera.h>
#include <opencv2/ccalib/omnidir.hpp>
namespace SuperVIO::Vision
{
    ///////////////////////////////////////////////////////////////
    Camera::ConstPtr Camera::
    Creat(std::string type,
          const std::vector<double>& input_intrinsic,
          const std::vector<double>& input_distortion,
          const std::vector<double>& output_intrinsic,
          const cv::Size& input_resolution,
          const cv::Size& output_resolution,
          bool use_gpu_for_undistortion)
    {
        transform(type.begin(), type.end(), type.begin(), ::tolower);
        ModelType model_type;
        if(type == "simplepinhole")
        {
            model_type = ModelType::SimplePinhole;
        }
        else if(type == "pinhole")
        {
            model_type = ModelType::Pinhole;
        }
        else if(type == "fisheye")
        {
            model_type = ModelType::FishEye;
        }
        else if(type == "omni")
        {
            model_type = ModelType::OMNI;
        }
        else
        {
            throw std::runtime_error("unsupport camera type");
        }

        return ConstPtr(new Camera(model_type, input_intrinsic, input_distortion,
                output_intrinsic, input_resolution, output_resolution, use_gpu_for_undistortion));
    }

    ///////////////////////////////////////////////////////////////
    Camera::
    Camera(ModelType _type,
           std::vector<double> input_intrinsic,
           std::vector<double> input_distortion,
           std::vector<double> output_intrinsic,
           const cv::Size& input_resolution,
           const cv::Size& output_resolution,
           bool use_gpu_for_undistortion):
            type_(_type),
            input_intrinsic_data_(std::move(input_intrinsic)),
            input_distortion_data_(std::move(input_distortion)),
            output_intrinsic_data_(std::move(output_intrinsic)),
            input_resolution_(input_resolution),
            output_resolution_(output_resolution),
            use_gpu_for_undistortion_(use_gpu_for_undistortion)
    {
        if(type_ != ModelType::SimplePinhole)
        {
            ROS_ASSERT(input_intrinsic_data_.size() >= 4);
            input_cv_intrinsic_matrix_ = ( cv::Mat_<double> ( 3,3 )
                    << input_intrinsic_data_[0], 0, input_intrinsic_data_[2],
                    0, input_intrinsic_data_[1], input_intrinsic_data_[3],
                    0, 0, 1);

            cv::Mat(cv::Size(input_distortion_data_.size(), 1), CV_64FC1, input_distortion_data_.data()).
                    copyTo(input_cv_distortion_vector_);
        }

        ROS_ASSERT(output_intrinsic_data_.size() == 4);
        output_cv_intrinsic_matrix_ = ( cv::Mat_<double> ( 3,3 )
                << output_intrinsic_data_[0], 0, output_intrinsic_data_[2],
                0, output_intrinsic_data_[1], output_intrinsic_data_[3],
                0, 0, 1);

        output_eigen_intrinsic_matrix_ << output_intrinsic_data_[0], 0, output_intrinsic_data_[2],
                0, output_intrinsic_data_[1], output_intrinsic_data_[3],
                0, 0, 1;

        switch (type_)
        {
            case ModelType::SimplePinhole:
            {
                ROS_ASSERT(input_resolution_ == output_resolution_);
                break;
            }
            case ModelType::Pinhole:
            {
                ROS_ASSERT(input_intrinsic_data_.size() == 4);
                std::set<size_t> distortion_model = {0, 4, 5, 8, 12, 14};
                size_t d = input_intrinsic_data_.size();
                auto iter = distortion_model.find(d);
                ROS_ASSERT(iter != distortion_model.end());
                cv::initUndistortRectifyMap(input_cv_intrinsic_matrix_,
                                            input_cv_distortion_vector_,
                                            cv::Mat::eye(3, 3, CV_64F),
                                            output_cv_intrinsic_matrix_,
                                            output_resolution_,
                                            CV_32FC1,
                                            map_x_,
                                            map_y_);
                break;
            }
            case ModelType::FishEye:
            {
                ROS_ASSERT(input_intrinsic_data_.size() == 4);
                ROS_ASSERT(input_distortion_data_.size() == 4);
                cv::fisheye::initUndistortRectifyMap(input_cv_intrinsic_matrix_,
                                                     input_cv_distortion_vector_,
                                                     cv::Mat::eye(3, 3, CV_64F),
                                                     output_cv_intrinsic_matrix_,
                                                     output_resolution_,
                                                     CV_32FC1,
                                                     map_x_,
                                                     map_y_);
                break;
            }
            case ModelType::OMNI:
            {
                ROS_ASSERT(input_intrinsic_data_.size() == 5);
                ROS_ASSERT(input_distortion_data_.size() == 4);
                cv::omnidir::initUndistortRectifyMap(input_cv_intrinsic_matrix_,
                                                     input_cv_distortion_vector_,
                                                     input_intrinsic_data_[4],
                                                     cv::Mat::eye(3, 3, CV_64F),
                                                     output_cv_intrinsic_matrix_,
                                                     output_resolution_,
                                                     CV_32FC1,
                                                     map_x_,
                                                     map_y_,
                                                     cv::omnidir::RECTIFY_PERSPECTIVE);

                break;
            }
        }
        if(type_ != ModelType::SimplePinhole && use_gpu_for_undistortion_)
        {
            map_x_gpu_.upload(map_x_);
            map_y_gpu_.upload(map_y_);
        }
    }

    ///////////////////////////////////////////////////////////////
    cv::Mat Camera::
    UndistortImage(const cv::Mat& distort_image) const
    {
        ROS_ASSERT(distort_image.size() == input_resolution_);
        cv::Mat undistort_image;
        if(type_ != ModelType::SimplePinhole)
        {
            if(use_gpu_for_undistortion_)
            {
                cv::cuda::GpuMat undistort_image_gpu, image_gpu(distort_image);
                cv::cuda::remap(image_gpu, undistort_image_gpu, map_x_gpu_, map_y_gpu_, cv::INTER_LINEAR);
                undistort_image_gpu.download(undistort_image);
            }
            else
            {
                cv::remap(distort_image, undistort_image, map_x_, map_y_, cv::INTER_LINEAR);
            }
        }
        else
        {
            undistort_image = distort_image.clone();
        }

        return undistort_image;
    }

    ///////////////////////////////////////////////////////////////
    Vector2 Camera::
    Project(const Vector3& camera_point) const
    {
        Vector3 c_point = camera_point / camera_point(2);

        double x = this->GetFx() * c_point.x() + this->GetCx();
        double y = this->GetFy() * c_point.y() + this->GetCy();

        return Vector2{x, y};
    }

    ///////////////////////////////////////////////////////////////
    Vector3 Camera::
    BackProject(const Vector2& image_point)const
    {
        double c_x = (image_point.x() - this->GetCx()) / this->GetFx();
        double c_y = (image_point.y() - this->GetCy()) / this->GetFy();

        return Vector3{c_x, c_y, 1};
    }
    ///////////////////////////////////////////////////////////////
    double Camera::
    GetFx() const
    {
        return output_intrinsic_data_[0];
    }

    ///////////////////////////////////////////////////////////////
    double Camera::
    GetFy() const
    {
        return output_intrinsic_data_[1];
    }

    ///////////////////////////////////////////////////////////////
    double Camera::
    GetCx() const
    {
        return output_intrinsic_data_[2];
    }

    ///////////////////////////////////////////////////////////////
    double Camera::
    GetCy() const
    {
        return output_intrinsic_data_[3];
    }

    ///////////////////////////////////////////////////////////////
    const Matrix3& Camera::
    GetIntrinsicMatrixEigen() const
    {
        return output_eigen_intrinsic_matrix_;
    }

    ///////////////////////////////////////////////////////////////
    const cv::Mat& Camera::
    GetIntrinsicMatrixCV() const
    {
        return output_cv_intrinsic_matrix_;
    }


    ///////////////////////////////////////////////////////////////
    cv::Size Camera::
    GetResolution() const
    {
        return output_resolution_;
    }
}//end of SuperVIO::Vision
