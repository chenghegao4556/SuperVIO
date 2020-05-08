//
// Created by chenghe on 3/9/20.
//

#ifndef SUPER_VIO_FEATURE_EXTRACTOR_H
#define SUPER_VIO_FEATURE_EXTRACTOR_H

#include <torch/script.h>
#include <torch/torch.h>

#include <vision/vision_measurements.h>


namespace SuperVIO::Vision
{
    class FeatureExtractor
    {
    public:
        typedef std::shared_ptr<FeatureExtractor> Ptr;
        //! SuperPoint inference parameters
        class Parameters
        {
        public:
            explicit Parameters(int _horizontal_boarder_width = 10,
                                int _vertical_boarder_width = 10,
                                const cv::Size& _inference_resolution = cv::Size(640, 480),
                                int _distance_threshold = 8,
                                double _confidence_threshold = 0.015f);


            int horizontal_boarder_width;
            int vertical_boarder_width;
            cv::Size inference_resolution;
            int distance_threshold;
            int cell_size;
            double confidence_threshold;
        };//end of Parameters

        /**
         * @brief factory
         * @param parameters
         * @param network_path
         * @return
         */
        static Ptr Creat(const Parameters& parameters,
                         const std::string& network_path);

        /**
         * @brief compute SuperPoint features
         */
        [[nodiscard]] FrameMeasurement Compute(const cv::Mat& input_image) const;
    protected:

        /**
         * @brief constructor
         */
        explicit FeatureExtractor(Parameters _parameters,
                                  const std::string& network_path);

        /**
         * @brief convert cv::Mat image to tensor
         */
        [[nodiscard]] torch::Tensor PreProcess(const cv::Mat& input_image) const;

        /**
         * @brief convert tensor to key points and descriptors
         */
        [[nodiscard]] FrameMeasurement PostProcess(const torch::Tensor& coarse_key_points,
                                                   const torch::Tensor& coarse_descriptors) const;

        /**
         * @brief suppress points with smaller confidence
         */
        [[nodiscard]] KeyPoints NonMaximumSuppression(const torch::Tensor& coarse_key_points) const;

    private:

        Parameters parameters_;
        std::shared_ptr<torch::jit::script::Module> module_;

    };//end of FeatureExtractor
}//end of SuperVIO



#endif //SUPER_VIO_FEATURE_EXTRACTOR_H
