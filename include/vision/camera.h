//
// Created by chenghe on 5/3/20.
//

#ifndef SUPER_VIO_CAMERA_H
#define SUPER_VIO_CAMERA_H
#include <ros/ros.h>
#include <memory>
#include <opencv2/ccalib.hpp>
#include <utility/eigen_type.h>
#include <opencv2/cudawarping.hpp>
namespace SuperVIO::Vision
{
    class Camera
    {
    public:
        typedef std::shared_ptr<Camera> Ptr;
        typedef std::shared_ptr<const Camera> ConstPtr;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        enum class ModelType
        {
            SimplePinhole = 0,
            Pinhole = 1,
            FishEye = 2,
            OMNI = 3,
        };

        static ConstPtr Creat(std::string type,
                              const std::vector<double>& input_intrinsic,
                              const std::vector<double>& input_distortion,
                              const std::vector<double>& output_intrinsic,
                              const cv::Size& input_resolution,
                              const cv::Size& output_resolution,
                              bool use_gpu_for_undistortion);

        /**
         * @brief undistort image point
         */
        [[nodiscard]] cv::Mat UndistortImage(const cv::Mat& distort_image) const;

        /**
         * @brief project camera point to image
         */
        [[nodiscard]] Vector2 Project(const Vector3& camera_point) const;

        /**
         * @brief back project image point to homogeneous plane
         */
        [[nodiscard]] Vector3 BackProject(const Vector2& image_point)const;

        //! get private data
        [[nodiscard]] double GetFx() const;
        [[nodiscard]] double GetFy() const;
        [[nodiscard]] double GetCx() const;
        [[nodiscard]] double GetCy() const;

        [[nodiscard]] const Matrix3& GetIntrinsicMatrixEigen() const;
        [[nodiscard]] const cv::Mat& GetIntrinsicMatrixCV() const;

        [[nodiscard]] cv::Size GetResolution() const;

    protected:
        /**
        * @brief constructor
        */
        Camera(ModelType _type,
               std::vector<double> input_intrinsic,
               std::vector<double> input_distortion,
               std::vector<double> output_intrinsic,
               const cv::Size& input_resolution,
               const cv::Size& output_resolution,
               bool use_gpu_for_undistortion);

    private:
        // camera type
        ModelType type_;

        //fx, fy, cx, cy optional(omni):xi
        std::vector<double> input_intrinsic_data_;
        std::vector<double> input_distortion_data_;
        std::vector<double> output_intrinsic_data_;

        cv::Size input_resolution_;
        cv::Size output_resolution_;

        bool use_gpu_for_undistortion_;

        cv::Mat input_cv_intrinsic_matrix_;
        cv::Mat input_cv_distortion_vector_;

        cv::Mat output_cv_intrinsic_matrix_;
        Matrix3 output_eigen_intrinsic_matrix_;

        cv::Mat map_x_;
        cv::Mat map_y_;
        cv::cuda::GpuMat map_x_gpu_;
        cv::cuda::GpuMat map_y_gpu_;
    };//end of Camera
}//end of SuperVIO::Vision
#endif //SUPER_VIO_CAMERA_H
