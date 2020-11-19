#include <dense_mapping/fast_bilateral_solver.h>
#include <ros/ros.h>
#include <ros/console.h>
namespace SuperVIO::DenseMapping
{
    typedef long long HashKey;
    /////////////////////////////////////////////////////////////////////////////////////////
    FastBilateralSolver::
    FastBilateralSolver(const cv::Mat& image, double sigma_spatial,
                        double sigma_luma, float lambda,
                        int max_solve_iteration, float error_tolerance):
        lambda_(lambda),
        max_solver_iteration_(max_solve_iteration),
        error_tolerance_(error_tolerance),
        cols_(image.cols),
        rows_(image.rows),
        num_pixels_(image.cols * image.rows),
        splat_indices_(num_pixels_)
    {
        CV_Assert(!image.empty()  && (image.depth() == CV_8U));
        std::vector<HashKey> hash_vector;
        for (int i = 0; i < 3; ++i)
        {
            hash_vector.push_back(static_cast<HashKey>(std::pow(255, i)));
        }
        std::unordered_map<HashKey, int> hashed_coordinates;
        hashed_coordinates.reserve(num_pixels_);
        int vertex_index = 0;
        int pixel_index = 0;
        // construct Splat(Slice) matrices
        for (int y = 0; y < rows_; ++y)
        {
            for (int x = 0; x < cols_; ++x)
            {
                const auto gray_value = static_cast<double>(image.at<uchar>(cv::Point(x, y)));
                std::vector<HashKey> coordinates{
                    static_cast<HashKey>(static_cast<double>(x) / sigma_spatial),
                    static_cast<HashKey>(static_cast<double>(y) / sigma_spatial),
                    static_cast<HashKey>(gray_value / sigma_luma)};

                // convert the coordinate to a hash value
                HashKey hash_coordinate = 0;
                for (int i = 0; i < 3; ++i)
                {
                    hash_coordinate += coordinates[i] * hash_vector[i];
                }

                // pixels whom are alike will have the same hash value.
                // We only want to keep a unique list of hash values, therefore make sure we only insert
                // unique hash values.
                auto iter = hashed_coordinates.find(hash_coordinate);
                if (iter == hashed_coordinates.end())
                {
                    hashed_coordinates.insert(std::pair(hash_coordinate, vertex_index));
                    splat_indices_[pixel_index] = vertex_index;
                    ++vertex_index;
                }
                else
                {
                    splat_indices_[pixel_index] = iter->second;
                }
                ++pixel_index;
            }
        }
        num_vertices_ = static_cast<int>(hashed_coordinates.size());


        // construct Blur matrices
        //std::chrono::steady_clock::time_point begin_blur_construction = std::chrono::steady_clock::now();
        blurs_ = Eigen::VectorXf::Ones(num_vertices_).asDiagonal();
        blurs_ *= 10.0f;
        std::vector<std::pair<int, int>> blur_indices;
        for(int offset = -1; offset <= 1;++offset)
        {
            if(offset == 0)
            {
                continue;
            }

            for (size_t i = 0; i < 3; ++i)
            {
                Eigen::SparseMatrix<float, Eigen::ColMajor> blur_temp(hashed_coordinates.size(),
                        hashed_coordinates.size());
                blur_temp.reserve(Eigen::VectorXi::Constant(num_vertices_,6));
                HashKey offset_hash_coordinate = offset * hash_vector[i];
                for (auto iter = hashed_coordinates.begin(); iter != hashed_coordinates.end(); ++iter)
                {
                    HashKey neighbor_coordinate = iter->first + offset_hash_coordinate;
                    auto iter_neighbor = hashed_coordinates.find(neighbor_coordinate);
                    if (iter_neighbor != hashed_coordinates.end())
                    {
                        blur_temp.insert(iter->second,iter_neighbor->second) = 1.0f;
                        blur_indices.emplace_back(iter->second, iter_neighbor->second);
                    }
                }
                blurs_ += blur_temp;
            }
        }
        blurs_.finalize();

        //bistochastize
        Eigen::VectorXf n = Eigen::VectorXf::Ones(num_vertices_);
        Eigen::VectorXf m = Eigen::VectorXf::Zero(num_vertices_);
        for (const auto& splat_index: splat_indices_)
        {
            m(splat_index) += 1.0f;
        }

        Eigen::VectorXf blured_n;
        for (int i = 0; i < 10; i++)
        {
            blured_n = Blur(n, blur_indices);
            n = ((n.array() * m.array()).array() / blured_n.array()).array().sqrt();
        }
        blured_n = Blur(n, blur_indices);

        m = n.array() * (blured_n).array();
        Dm_ = m.asDiagonal();
        Dn_ = n.asDiagonal();
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    cv::Mat FastBilateralSolver::
    Filter(const cv::Mat& raw_depth_map, const cv::Mat& confidence_map) const
    {
        CV_Assert(!raw_depth_map.empty()  && (raw_depth_map.depth() == CV_32F)  && raw_depth_map.channels() == 1);
        CV_Assert(!confidence_map.empty() && (confidence_map.depth() == CV_32F) && confidence_map.channels()==1);
        Eigen::VectorXf x = Eigen::VectorXf::Zero(num_pixels_);
        Eigen::VectorXf w = Eigen::VectorXf::Zero(num_pixels_);

        for(int i = 0; i < cols_; ++i)
        {
            for(int j = 0; j < rows_; ++j)
            {
                int index = j * cols_ + i;
                x(index) = raw_depth_map.at<float>(cv::Point(i, j));
                w(index) = confidence_map.at<float>(cv::Point(i, j));
            }
        }

        Eigen::VectorXf w_splat = Splat(w, splat_indices_);
        Eigen::SparseMatrix<float, Eigen::ColMajor> A_data(num_vertices_, num_vertices_);
        Eigen::SparseMatrix<float, Eigen::ColMajor> A(num_vertices_, num_vertices_);
        A_data  = w_splat.asDiagonal();
        A = lambda_ * (Dm_ - Dn_ * (blurs_ * Dn_)) + A_data;

        //construct b
        Eigen::VectorXf b = Eigen::VectorXf::Zero(num_vertices_);
        Eigen::VectorXf y0 = Eigen::VectorXf::Zero(num_vertices_);
        Eigen::VectorXf y1 = Eigen::VectorXf::Zero(num_vertices_);
        for (size_t i = 0; i < splat_indices_.size(); i++)
        {
            b(splat_indices_[i])  += x(i) * w(i);
            y0(splat_indices_[i]) += x(i);
            y1(splat_indices_[i]) += 1.0f;
        }
        for (int i = 0; i < num_vertices_; i++)
        {
            y0(i) = y0(i)/y1(i);
        }
        b = b.unaryExpr([](float v)   { return (std::isinf(v) || std::isnan(v))? 0.0f : v; });
        y0 = y0.unaryExpr([](float v) { return (std::isinf(v) || std::isnan(v))? 0.0f : v; });
        for (int k=0; k < A.outerSize(); ++k)
        {
            for (Eigen::SparseMatrix<float, Eigen::ColMajor>::InnerIterator it(A,k); it; ++it)
            {
                if(std::isinf(it.value()) || std::isnan(it.value()))
                {
                    it.valueRef() = 0.0f;
                }
            }
        }

        // solve Ay = b
        Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower|Eigen::Upper> cg;
        cg.compute(A);
        cg.setMaxIterations(max_solver_iteration_);
        cg.setTolerance(error_tolerance_);
        // y = cg.solve(b);
        Eigen::VectorXf y = cg.solveWithGuess(b,y0);
        std::cout<<"estimated error: " << cg.error()<<std::endl;

        //slice
        cv::Mat refine_depth_map = cv::Mat::zeros(raw_depth_map.size(), CV_32FC1);
        for(int i = 0; i < cols_; ++i)
        {
            for(int j = 0; j < rows_; ++j)
            {
                int index = j * cols_ + i;
                refine_depth_map.at<float>(cv::Point(i, j)) = y(splat_indices_[index]);
            }
        }

        return refine_depth_map;
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    Eigen::VectorXf FastBilateralSolver::
    Splat(const Eigen::VectorXf& input, const std::vector<int>& splat_indices)const
    {
        Eigen::VectorXf output = Eigen::VectorXf::Zero(num_vertices_);
        for (size_t i = 0; i < splat_indices.size(); i++)
        {
            output(splat_indices[i]) += input(i);
        }

        return output;
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    Eigen::VectorXf FastBilateralSolver::
    Blur(const Eigen::VectorXf& input, const std::vector<std::pair<int, int>>& blur_indices)const
    {
        Eigen::VectorXf output = input * 10.0f;
        for (const auto & blur_index : blur_indices)
        {
            output(blur_index.first) += input(blur_index.second);
        }

        return output;
    }
}