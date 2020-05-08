//
// Created by chenghe on 3/9/20.
//
#include <vision/feature_extractor.h>

#include <utility>

#include <memory>

namespace SuperVIO::Vision
{
    ///////////////////////////////////////////////////////////////////////////////////////
    FeatureExtractor::Parameters::
    Parameters(int _horizontal_boarder_width,
               int _vertical_boarder_width,
               const cv::Size& _inference_resolution,
               int _distance_threshold,
               double _confidence_threshold):
               horizontal_boarder_width(_horizontal_boarder_width),
               vertical_boarder_width(_vertical_boarder_width),
               inference_resolution(_inference_resolution),
               distance_threshold(_distance_threshold),
               cell_size(8),
               confidence_threshold(_confidence_threshold)
    {

    }

    ///////////////////////////////////////////////////////////////////////////////////////
    FeatureExtractor::Ptr FeatureExtractor::
    Creat(const Parameters& parameters,
          const std::string& network_path)
    {
        Ptr ptr(new FeatureExtractor(parameters, network_path));

        return ptr;
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    FeatureExtractor::
    FeatureExtractor(Parameters _parameters,
                     const std::string& network_path):
                        parameters_(std::move(_parameters))
    {
        module_ = torch::jit::load(network_path);
        module_->to(at::kCUDA);
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    FrameMeasurement FeatureExtractor::
    Compute(const cv::Mat& input_image) const
    {
        float horizontal_scale = static_cast<float>(input_image.size().width)/
                static_cast<float>(parameters_.inference_resolution.width);
        float vertical_scale = static_cast<float>(input_image.size().height)/
                static_cast<float>(parameters_.inference_resolution.height);
        torch::Tensor input_tensor = this->PreProcess(input_image);
        auto output = module_->forward({input_tensor}).toTuple();

        torch::Tensor coarse_key_points  = output->elements()[0].toTensor().to(torch::kCPU).squeeze();
        torch::Tensor coarse_descriptors = output->elements()[1].toTensor();

        auto features = this->PostProcess(coarse_key_points, coarse_descriptors);
        for(auto& feature: features.key_points)
        {
            feature.point.x *= horizontal_scale;
            feature.point.y *= vertical_scale;
            feature.sigma_x *= horizontal_scale;
            feature.sigma_y *= vertical_scale;
            feature.sigma_x = feature.sigma_x < 1.0 ? 1 : feature.sigma_x;
            feature.sigma_y = feature.sigma_y < 1.0 ? 1 : feature.sigma_y;
        }

        return features;
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    torch::Tensor FeatureExtractor::
    PreProcess(const cv::Mat& input_image) const
    {
        cv::Mat processed_image;
        if(input_image.channels() == 3)
        {
            cv::cvtColor( input_image,  processed_image, cv::COLOR_BGR2GRAY);
        }
        else
        {
            processed_image = input_image;
        }

        cv::resize(processed_image, processed_image, parameters_.inference_resolution);

        processed_image.convertTo(processed_image, CV_32F, 1.0 / 255);
        std::vector<int64_t> dims = {1, parameters_.inference_resolution.height,
                                     parameters_.inference_resolution.width, 1};
        auto output_tensor = torch::from_blob(processed_image.data, dims, torch::kFloat32);

        output_tensor = output_tensor.permute({0,3,1,2});
        output_tensor = output_tensor.to(torch::kCUDA);

        return output_tensor;
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    FrameMeasurement FeatureExtractor::
    PostProcess(const torch::Tensor& coarse_key_points,
                const torch::Tensor& coarse_descriptors) const
    {
        auto dense_tensor = coarse_key_points.exp();
        dense_tensor = dense_tensor.div(dense_tensor.sum(0).add(0.00001));
        auto no_dust_tensor = dense_tensor.slice(0, 0, -1);

        int Wc = parameters_.inference_resolution.width / parameters_.cell_size;
        int Hc = parameters_.inference_resolution.height / parameters_.cell_size;

        no_dust_tensor =no_dust_tensor.permute({1, 2, 0});

        auto heatmap = no_dust_tensor.reshape({Hc, Wc,parameters_.cell_size,
                                               parameters_.cell_size});
        heatmap = heatmap.permute({0, 2, 1, 3});
        heatmap = heatmap.reshape({parameters_.inference_resolution.height,
                                   parameters_.inference_resolution.width});

        auto mask = heatmap.gt(parameters_.confidence_threshold);
        auto key_points_tensor = mask.nonzero();
        auto xs = key_points_tensor.slice(1, 0, 1).squeeze().to(torch::kFloat32);
        auto ys = key_points_tensor.slice(1, 1).squeeze().to(torch::kFloat32);
        auto confidence_map = torch::masked_select(heatmap, mask);
        key_points_tensor = torch::stack({ys, xs, confidence_map},1);

        auto key_points = this->NonMaximumSuppression(key_points_tensor);

        std::vector<int64_t> dims = {static_cast<int64_t>(key_points.size()), 2};
        key_points_tensor = torch::zeros(dims).to(torch::kFloat32);
        int i = 0;
        for(const auto& key_point: key_points)
        {
            key_points_tensor[i][0] = key_point.point.x;
            key_points_tensor[i][1] = key_point.point.y;
            ++i;
        }
        torch::Tensor div = torch::zeros({2}).to(torch::kFloat);
        div[0] = static_cast<float>(parameters_.inference_resolution.width) / 2.0f;
        div[1] = static_cast<float>(parameters_.inference_resolution.height) / 2.0f;
        auto sample_key_points = key_points_tensor.div(div).sub(1).contiguous();
        sample_key_points = sample_key_points.view({1 ,1, -1, 2});
        sample_key_points = sample_key_points.cuda();
        int D = coarse_descriptors.size(1);

        auto descriptors_tensor = torch::cudnn_grid_sampler(coarse_descriptors, sample_key_points);
        descriptors_tensor = descriptors_tensor.cpu().reshape({D,-1});
        descriptors_tensor = descriptors_tensor.div(torch::norm(descriptors_tensor, 2, {0}));
        descriptors_tensor = descriptors_tensor.t().contiguous();

        cv::Mat descriptors;
        cv::Mat(cv::Size(256,descriptors_tensor.size(0)), CV_32FC1, descriptors_tensor.data<float>()).copyTo(descriptors);

        return FrameMeasurement(key_points, descriptors);
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    KeyPoints FeatureExtractor::
    NonMaximumSuppression(const torch::Tensor& coarse_key_points) const
    {
        torch::Tensor confidences_map = coarse_key_points.slice(1,2);
        torch::Tensor sort_indices = confidences_map.argsort(0,true).squeeze();
        auto key_points_tensor = coarse_key_points.index_select(0, sort_indices);

        cv::Mat key_points_mat (cv::Size(3, key_points_tensor.size(0)), CV_32FC1,
                               key_points_tensor.data<float>());
        cv::Mat grid = cv::Mat(parameters_.inference_resolution, CV_8UC1);
        cv::Mat indices = cv::Mat(parameters_.inference_resolution, CV_16UC1);
        grid.setTo(0);
        indices.setTo(0);
        int length = key_points_mat.rows;
        for (int i = 0; i < length; i++)
        {
            int x = (int)key_points_mat.at<float>(i,0);
            int y = (int)key_points_mat.at<float>(i,1);
            grid.at<char>(y, x) = 1;
            indices.at<unsigned short>(y, x) = i;
        }

        int min_x = parameters_.distance_threshold + parameters_.horizontal_boarder_width;
        int min_y = parameters_.distance_threshold + parameters_.vertical_boarder_width;
        int max_x = parameters_.inference_resolution.width + parameters_.distance_threshold -
                parameters_.horizontal_boarder_width;
        int max_y = parameters_.inference_resolution.height + parameters_.distance_threshold -
                parameters_.vertical_boarder_width;


        cv::copyMakeBorder(grid, grid, parameters_.distance_threshold, parameters_.distance_threshold,
                           parameters_.distance_threshold, parameters_.distance_threshold,
                           cv::BORDER_CONSTANT, 0);
        cv::copyMakeBorder(indices, indices, parameters_.distance_threshold, parameters_.distance_threshold,
                parameters_.distance_threshold, parameters_.distance_threshold, cv::BORDER_CONSTANT, 0);
        //suppression
        std::vector<KeyPoint> key_points;
        for(int i = 0; i < length; ++i)
        {
            int x = (int)key_points_mat.at<float>(i, 0) + parameters_.distance_threshold;
            int y = (int)key_points_mat.at<float>(i, 1) + parameters_.distance_threshold;
            float max_confidence = key_points_mat.at<float>(i, 3);
            if(grid.at<char>(y, x) == 1)
            {

                std::vector<float> xs;
                std::vector<float> ys;
                std::vector<float> confidences;
                for(int k = -parameters_.distance_threshold; k < (parameters_.distance_threshold + 1); ++k)
                {
                    for(int j = -parameters_.distance_threshold; j < (parameters_.distance_threshold + 1); ++j)
                    {
                        if(grid.at<char>(y + k, x + j) == 1)
                        {
                            if(std::abs(k) < 5 && std::abs(j) < 5)
                            {
                                xs.push_back(x + j - parameters_.distance_threshold);
                                ys.push_back(y + k - parameters_.distance_threshold);
                                confidences.push_back(key_points_mat.at<float>(i, 2));
                            }
                        }
                        grid.at<char>(y+k, x+j) = 0;
                    }
                }
                float sum_confidence = std::accumulate(confidences.begin(), confidences.end(), 0.0f);
                float sum_x = 0;
                float sum_y = 0;
                for(size_t index = 0; index < xs.size(); ++index)
                {
                    sum_x += xs[index] * confidences[index];
                    sum_y += ys[index] * confidences[index];
                }
                float avg_x = sum_x / sum_confidence;
                float avg_y = sum_y / sum_confidence;
                float variance_x = 0;
                float variance_y = 0;
                std::for_each(xs.begin(), xs.end(),
                        [&](float x){variance_x += std::pow(x - avg_x, 2);});
                std::for_each(ys.begin(), ys.end(),
                              [&](float y){variance_y += std::pow(y - avg_y, 2);});

                float sigma_x = std::sqrt(variance_x / static_cast<float>(xs.size()));
                float sigma_y = std::sqrt(variance_y / static_cast<float>(ys.size()));

                if (x <= min_x || y <= min_y || y >= max_y || x >= max_x)
                {
                    grid.at<char>(y, x) = 0;
                }
                else
                {
                    grid.at<char>(y, x) = -1;
                    key_points.emplace_back(cv::Point2f(avg_x, avg_y), sigma_x, sigma_y, max_confidence);
                }
            }
        }

        return key_points;
    }
}//end of Vision
