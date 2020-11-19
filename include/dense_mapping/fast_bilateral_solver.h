

#ifndef SUPER_VIO_FAST_BILATERAL_SOLVER_H
#define SUPER_VIO_FAST_BILATERAL_SOLVER_H

#ifdef EIGEN_MPL2_ONLY
#undef EIGEN_MPL2_ONLY
#endif


#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>

#include<opencv2/core/core.hpp>
#include<opencv2/core/eigen.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

#include <cmath>
#include <chrono>
#include <vector>
#include <memory>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <unordered_map>
namespace SuperVIO::DenseMapping
{
    class FastBilateralSolver
    {
    public:
        explicit FastBilateralSolver(const cv::Mat& image, double sigma_spatial = 16,
                                     double sigma_luma = 16, float lambda = 128,
                                     int max_solve_iteration = 25, float error_tolerance = 1e-5);

        [[nodiscard]] cv::Mat Filter(const cv::Mat& raw_depth_map, const cv::Mat& confidence_map) const;

    protected:
        [[nodiscard]] Eigen::VectorXf
        Splat(const Eigen::VectorXf& input, const std::vector<int>& splat_indices) const;

        [[nodiscard]] Eigen::VectorXf
        Blur(const Eigen::VectorXf& input, const std::vector<std::pair<int, int>>& blur_indices) const;

    private:
        float lambda_;
        size_t max_solver_iteration_;
        float error_tolerance_;
        int cols_;
        int rows_;
        int num_pixels_;
        std::vector<int> splat_indices_;
        int num_vertices_;
        Eigen::SparseMatrix<float, Eigen::ColMajor> blurs_;
        Eigen::SparseMatrix<float, Eigen::ColMajor> Dn_;
        Eigen::SparseMatrix<float, Eigen::ColMajor> Dm_;
    };
}


#endif //SUPER_VIO_FAST_BILATERAL_SOLVER_H