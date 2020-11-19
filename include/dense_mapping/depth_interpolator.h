//
// Created by chenghe on 8/2/20.
//

#ifndef SUPER_VIO_DEPTH_INTERPOLATOR_H
#define SUPER_VIO_DEPTH_INTERPOLATOR_H
#include <Eigen/Sparse>
#include <utility/eigen_type.h>
#include <utility/eigen_base.h>
#include <vision/camera.h>
#include <opencv2/optflow.hpp>
#include <opencv2/imgproc/imgproc.hpp>
namespace SuperVIO::DenseMapping
{
    class DepthInterpolator
    {
    public:
        class Parameters
        {
        public:
            explicit Parameters(double _canny_thresh_1 = 50,
                                double _canny_thresh_2 = 200,
                                int _k_I = 5,
                                int _k_T = 7,
                                int _k_F = 31,
                                double _lambda_d = 1,
                                double _lambda_t = 0.01,
                                double _lambda_s = 1,
                                int _num_solver_iterations = 400);

            double canny_thresh_1;
            double canny_thresh_2;
            int k_I;
            int k_T;
            int k_F;
            double lambda_d;
            double lambda_t;
            double lambda_s;
            int num_solver_iterations;

        };
        static cv::Mat Estimate(const cv::Mat& sparse_points,
                                const std::vector<cv::Mat>& reference_images,
                                const cv::Mat& current_image,
                                const cv::Mat& last_depth_map,
                                const Parameters& parameters = Parameters());
    protected:

        static cv::Mat Solve(const cv::Mat& sparse_points,
                             const cv::Mat& hard_edges,
                             const cv::Mat& soft_edges,
                             const cv::Mat& last_depth_map,
                             const Parameters& parameters);

        static cv::Mat GetInitialization(const cv::Mat& sparse_points,
                                         const cv::Mat& last_depth_map);

        static cv::Mat ProjectDepthMap(const cv::Mat& _map_0,     const Matrix3& K,
                                       const Quaternion& q_w_i_0, const Vector3& p_w_i_0,
                                       const Quaternion& q_w_i_1, const Vector3& p_w_i_1,
                                       const Quaternion& q_i_c,   const Vector3& p_i_c);

        static std::vector<cv::Mat> EstimateFlow(const std::vector<cv::Mat>& reference_images,
                                                 const cv::Mat& current_image);

        static cv::Mat EstimateSoftEdges(const cv::Mat& image, const std::vector<cv::Mat>& flows,
                                         const Parameters& parameters);

        static cv::Mat EstimateHardEdges(const cv::Mat& image, const Parameters& parameters);

        static cv::Mat EstimateGradientMagnitude(const cv::Mat& img_grad_x,
                                                 const cv::Mat& img_grad_y);

        static std::pair<cv::Mat, cv::Mat>
        EstimateFlowGradientMagnitude(const cv::Mat& flow,
                                      const cv::Mat& img_grad_x, const cv::Mat& img_grad_y);

        static std::pair<cv::Mat, cv::Mat> GetImageGradient(const cv::Mat& image);
    };//end of DepthInterpolator
}//end of SuperVIO::DenseMapping

#endif //SUPER_VIO_DEPTH_INTERPOLATOR_H
