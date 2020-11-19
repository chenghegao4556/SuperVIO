//
// Created by chenghe on 8/2/20.
//
#include <dense_mapping/depth_interpolator.h>
#include <queue>
namespace SuperVIO::DenseMapping
{
    /////////////////////////////////////////////////////////////////////////////////////////
    DepthInterpolator::Parameters::
    Parameters(double _canny_thresh_1,
               double _canny_thresh_2,
               int _k_I,
               int _k_T,
               int _k_F,
               double _lambda_d,
               double _lambda_t,
               double _lambda_s,
               int _num_solver_iterations):
        canny_thresh_1(_canny_thresh_1),
        canny_thresh_2(_canny_thresh_2),
        k_I(_k_I),
        k_T(_k_T),
        k_F(_k_F),
        lambda_d(_lambda_d),
        lambda_t(_lambda_t),
        lambda_s(_lambda_s),
        num_solver_iterations(_num_solver_iterations)
    {

    }

    /////////////////////////////////////////////////////////////////////////////////////////
    cv::Mat DepthInterpolator::
    Estimate(const cv::Mat& sparse_points,
             const std::vector<cv::Mat>& reference_images,
             const cv::Mat& current_image,
             const cv::Mat& last_depth_map,
             const Parameters& parameters)
    {
        ROS_INFO_STREAM("ESTIMATE FLOWS");
        auto flows = EstimateFlow(reference_images, current_image);
        ROS_INFO_STREAM("ESTIMATE SOFT EDGE");
        auto soft_edges = EstimateSoftEdges(current_image, flows, parameters);
        ROS_INFO_STREAM("ESTIMATE HARD EDGE");
        auto hard_edges = EstimateHardEdges(current_image, parameters);
        ROS_INFO_STREAM("SOLVE");
        auto current_depth = Solve(sparse_points, hard_edges, soft_edges, last_depth_map, parameters);

        return current_depth;
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    cv::Mat DepthInterpolator::
    Solve(const cv::Mat& sparse_points,
          const cv::Mat& hard_edges,
          const cv::Mat& soft_edges,
          const cv::Mat& last_depth_map,
          const Parameters& parameters)
    {
        int w = sparse_points.cols;
        int h = sparse_points.rows;
        int num_pixels = w * h;

        Eigen::SparseMatrix<float> A(num_pixels * 3, num_pixels);
        Eigen::VectorXf b = Eigen::VectorXf::Zero(num_pixels * 3);
        Eigen::VectorXf x0 = Eigen::VectorXf::Zero(num_pixels);
        int num_entries = 0;

        cv::Mat smoothness = cv::max(1 - soft_edges, 0);
        cv::Mat smoothness_x = cv::Mat::zeros(cv::Size(w, h), CV_64FC1);
        cv::Mat smoothness_y = cv::Mat::zeros(cv::Size(w, h), CV_64FC1);

        cv::Mat initialization = GetInitialization(sparse_points, last_depth_map);

        std::vector<Eigen::Triplet<float>> tripletList;

        for (int y = 1; y < h - 1; y++)
        {
            for (int x = 1; x < w - 1; x++)
            {
                int idx = x + y * w;
                x0(idx) = initialization.at<double>(y, x);
                if (sparse_points.at<double>(y, x) > 0.00)
                {
                    tripletList.emplace_back(
                            Eigen::Triplet<float>(num_entries, idx, parameters.lambda_d));
                    b(num_entries) = (1.0 / sparse_points.at<double>(y, x)) * parameters.lambda_d;
                    num_entries++;
                }
                else if (!last_depth_map.empty() &&
                         last_depth_map.at<double>(y, x) > 0)
                {
                    tripletList.emplace_back(
                            Eigen::Triplet<float>(num_entries, idx, parameters.lambda_t));
                    b(num_entries) = (1.0 / last_depth_map.at<double>(y, x)) * parameters.lambda_t;
                    num_entries++;
                }

                double smoothness_weight =
                        parameters.lambda_s * std::min(smoothness.at<double>(y, x),
                                                       smoothness.at<double>(y - 1, x));

                if (hard_edges.at<double>(y, x) == hard_edges.at<double>(y - 1, x))
                {
                    smoothness_x.at<double>(y, x) = smoothness_weight;
                    tripletList.emplace_back(
                            Eigen::Triplet<float>(num_entries, idx - w, smoothness_weight));
                    tripletList.emplace_back(
                            Eigen::Triplet<float>(num_entries, idx, -smoothness_weight));
                    b(num_entries) = 0;
                    num_entries++;
                }

                smoothness_weight = parameters.lambda_s * std::min(smoothness.at<double>(y, x),
                                                                   smoothness.at<double>(y, x - 1));

                if (hard_edges.at<double>(y, x) == hard_edges.at<double>(y, x - 1))
                {
                    smoothness_y.at<double>(y, x) = smoothness_weight;
                    tripletList.emplace_back(
                            Eigen::Triplet<float>(num_entries, idx - 1, smoothness_weight));
                    tripletList.emplace_back(
                            Eigen::Triplet<float>(num_entries, idx, -smoothness_weight));
                    b(num_entries) = 0;
                    num_entries++;
                }
            }
        }

        A.setFromTriplets(tripletList.begin(), tripletList.end());

        Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower | Eigen::Upper> cg;

        cg.compute(A.transpose() * A);
        cg.setMaxIterations(parameters.num_solver_iterations);
        cg.setTolerance(1e-05);
        Eigen::VectorXf x_vec = cg.solveWithGuess(A.transpose() * b, x0);

        cv::Mat depth = cv::Mat::zeros(h, w, CV_64FC1);
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                depth.at<double>(y, x) = 1.0 / (x_vec(x + y * w) + 1e-7);
            }
        }

        return depth;
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    cv::Mat DepthInterpolator::
    GetInitialization(const cv::Mat& sparse_points,
                      const cv::Mat& last_depth_map)
    {
        cv::Mat initialization = sparse_points.clone();
        if (!last_depth_map.empty())
        {
            cv::Mat inv = 1.0 / last_depth_map;
            inv.copyTo(initialization, last_depth_map > 0);
        }

        int h = sparse_points.rows;
        int w = sparse_points.cols;
        double last_known = -1;
        double first_known = -1;

        double min, max;
        cv::Point min_loc, max_loc;
        cv::minMaxLoc(sparse_points, &min, &max, &min_loc, &max_loc);

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                if (sparse_points.at<double>(y, x) > 0)
                {
                    last_known = 1.0 / sparse_points.at<double>(y, x);
                }
                else if (initialization.at<double>(y, x) > 0)
                {
                    last_known = initialization.at<double>(y, x);
                }
                if (first_known < 0)
                {
                    first_known = last_known;
                }
                initialization.at<double>(y, x) = last_known;
            }
        }

        cv::Mat first_known_mat =
                cv::Mat::ones(h, w, initialization.type()) * first_known;
        cv::Mat mask = initialization < 0;
        first_known_mat.copyTo(initialization, mask);

        return initialization;
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    cv::Mat DepthInterpolator::
    ProjectDepthMap(const cv::Mat& depth_map_0,     const Matrix3& K,
                    const Quaternion& q_w_i_0, const Vector3& p_w_i_0,
                    const Quaternion& q_w_i_1, const Vector3& p_w_i_1,
                    const Quaternion& q_i_c,   const Vector3& p_i_c)
    {
        const Quaternion q_w_c_0 = q_w_i_0 * q_i_c;
        const Vector3    p_w_c_0 = q_w_i_0 * p_i_c + p_w_i_0;

        const Quaternion q_w_c_1 = q_w_i_1 * q_i_c;
        const Vector3    p_w_c_1 = q_w_i_1 * p_i_c + p_w_i_1;

        const Quaternion q_1_0 = q_w_c_1.inverse() * q_w_c_0;
        const Vector3    p_1_0 = q_w_c_1.inverse() * (p_w_c_0 - p_w_c_1);

        const Matrix3 K_inv = K.inverse();

        Eigen::Matrix<double, 3, 4> T_1_0;
        T_1_0.leftCols<3>()  = q_1_0.matrix();
        T_1_0.rightCols<1>() = p_1_0;

        cv::Mat depth_map_1 = cv::Mat::zeros(depth_map_0.size(), CV_64FC1);
        cv::Rect rect(0, 0, depth_map_1.size().width, depth_map_1.size().height);
        for(int x = 0; x < depth_map_0.cols; ++x)
        {
            for(int y = 0; y < depth_map_0.rows; ++y)
            {
                const auto& depth = depth_map_0.at<double>(cv::Point(x,y));
                if(depth <= 0)
                {
                    continue;
                }
                const Vector4 point_0 = (depth * K_inv * Vector3(x, y, 1)).homogeneous();
                const Vector3 point_1 = T_1_0 * point_0;
                const double new_depth = point_1(2);
                const Vector2 projection = (K * (point_1/point_1(2))).head<2>();
                if(rect.contains(cv::Point(projection(0), projection(1))))
                {
                    depth_map_1.at<double>(cv::Point(projection(0), projection(1))) = new_depth;
                }
            }
        }

        return depth_map_1;
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    std::vector<cv::Mat> DepthInterpolator::
    EstimateFlow(const std::vector<cv::Mat>& reference_images,
                 const cv::Mat& current_image)
    {
        std::vector<cv::Mat> flows;
        const auto dis = cv::optflow::createOptFlow_DIS(2);
        for(const auto& ref: reference_images)
        {
            cv::Mat flow;
            dis->calc(current_image, ref, flow);
            flows.push_back(flow);
        }

        return flows;
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    cv::Mat DepthInterpolator::
    EstimateSoftEdges(const cv::Mat& image, const std::vector<cv::Mat>& flows, const Parameters& parameters)
    {
        std::pair<cv::Mat, cv::Mat> img_grad = GetImageGradient(image);
        cv::Mat img_grad_magnitude =
                EstimateGradientMagnitude(img_grad.first, img_grad.second);

        cv::Mat flow_gradient_magnitude =
                cv::Mat::zeros(img_grad_magnitude.size(), img_grad_magnitude.depth());
        cv::Mat max_reliability =
                cv::Mat::zeros(img_grad_magnitude.size(), img_grad_magnitude.depth());

        int height = flows[0].rows;
        int width = flows[0].cols;

        for (const auto& flow : flows)
        {
            std::pair<cv::Mat, cv::Mat> FlowGradMag =
                    EstimateFlowGradientMagnitude(flow, img_grad.first, img_grad.second);
            cv::Mat magnitude = FlowGradMag.first;
            cv::Mat reliability = FlowGradMag.second;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    if (reliability.at<float>(y, x) > max_reliability.at<float>(y, x))
                    {
                        flow_gradient_magnitude.at<double>(y, x) = magnitude.at<double>(y, x);
                    }
                }
            }
        }

        cv::GaussianBlur(flow_gradient_magnitude, flow_gradient_magnitude,
                         cv::Size(parameters.k_F, parameters.k_F), 0);
        flow_gradient_magnitude = flow_gradient_magnitude.mul(img_grad_magnitude);
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(flow_gradient_magnitude, &minVal, &maxVal, &minLoc, &maxLoc);
        flow_gradient_magnitude /= maxVal;

        return flow_gradient_magnitude;

    }

    /////////////////////////////////////////////////////////////////////////////////////////
    cv::Mat DepthInterpolator::
    EstimateHardEdges(const cv::Mat& image, const Parameters& parameters)
    {
        cv::Mat edges;
        cv::Canny(image, edges, parameters.canny_thresh_1, parameters.canny_thresh_2);
        edges.convertTo(edges, CV_64FC1);

        return edges;
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    std::pair<cv::Mat, cv::Mat> DepthInterpolator::
    EstimateFlowGradientMagnitude(const cv::Mat& flow,
                                  const cv::Mat& img_grad_x, const cv::Mat& img_grad_y)
    {
        std::vector<cv::Mat> grad_x, grad_y;
        cv::Mat tmp_x, tmp_y;
        cv::Sobel(flow, tmp_x, CV_64F, 1, 0, 5);
        cv::Sobel(flow, tmp_y, CV_64F, 0, 1, 5);
        cv::split(tmp_x, grad_x);
        cv::split(tmp_y, grad_y);

        cv::Mat flow_grad_x = cv::max(grad_x[0], grad_x[1]);
        cv::Mat flow_grad_y = cv::max(grad_y[0], grad_y[1]);

        cv::Mat flow_grad_magnitude;
        cv::sqrt(flow_grad_x.mul(flow_grad_x) + flow_grad_y.mul(flow_grad_y),
                 flow_grad_magnitude);

        cv::Mat reliability = cv::Mat::zeros(flow.size(), flow.depth());

        int height = img_grad_x.rows;
        int width = img_grad_x.cols;

        for (int y = 0; y < height; y++)
        {
            for (int x = 1; x < width; x++)
            {
                Vector2 gradient_dir(img_grad_y.at<double>(y, x),
                                     img_grad_x.at<double>(y, x));
                if (gradient_dir.norm() == 0)
                {
                    reliability.at<float>(y, x) = 0;
                    continue;
                }
                gradient_dir /= gradient_dir.norm();
                Vector2 center_pixel(y, x);
                Vector2 p0 = center_pixel + gradient_dir;
                Vector2 p1 = center_pixel - gradient_dir;

                if (p0[0] < 0 || p1[0] < 0 || p0[1] < 0 || p1[1] < 0 || p0[0] >= height ||
                    p0[1] >= width || p1[0] >= height || p1[1] >= height)
                {
                    reliability.at<float>(y, x) = -100;
                    continue;
                }

                Vector2 flow_p0(flow.at<cv::Vec2f>(int(p0[0]), int(p0[1]))[0],
                                flow.at<cv::Vec2f>(int(p0[0]), int(p0[1]))[1]);
                Vector2 flow_p1(flow.at<cv::Vec2f>(int(p1[0]), int(p1[1]))[0],
                                flow.at<cv::Vec2f>(int(p1[0]), int(p1[1]))[1]);

                double f0 = flow_p0.dot(gradient_dir);
                double f1 = flow_p1.dot(gradient_dir);
                reliability.at<float>(y, x) = f1 - f0;
            }
        }

        return {flow_grad_magnitude, reliability};
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    cv::Mat DepthInterpolator::
    EstimateGradientMagnitude(const cv::Mat& img_grad_x,
                              const cv::Mat& img_grad_y)
    {
        cv::Mat img_grad_magnitude;
        cv::sqrt(img_grad_x.mul(img_grad_x) + img_grad_y.mul(img_grad_y),
                 img_grad_magnitude);

        return img_grad_magnitude;
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    std::pair<cv::Mat, cv::Mat> DepthInterpolator::
    GetImageGradient(const cv::Mat& image)
    {
        cv::Mat grad_x, grad_y;

        cv::Sobel(image, grad_x, CV_64F, 1, 0, 5);
        cv::Sobel(image, grad_y, CV_64F, 0, 1, 5);

        return {grad_x, grad_y};
    }
}//end of SuperVIO::DenseMapping
