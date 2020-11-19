//
// Created by chenghe on 5/18/20.
//
#include <loop_closure/image_retrieval.h>
#include <loop_closure/internal/io.h>
using namespace SuperVIO::LoopClosure;

int main()
{
    SuperVIO::Vision::FeatureExtractor::Parameters params;
    auto fe = SuperVIO::Vision::FeatureExtractor::Creat(params, "/home/chenghe/catkin_ws/src/SuperVIO/data/superpoint.pt");
    std::vector<std::string> train_image_names, query_image_names;
    Internal::GetImageNames("/media/chenghe/ChengheGao/kitti odometry/data_odometry_color/dataset/sequences/02/image_2",
                            train_image_names);
    Internal::GetImageNames("/media/chenghe/ChengheGao/kitti odometry/data_odometry_color/dataset/sequences/02/query",
                            query_image_names);
    DBoW3::Vocabulary vocabulary("./vocabulary.yml.gz");
    DBoW3::Database database(vocabulary, false, 0);
    std::vector<cv::Mat> train_images;
    for(size_t i = 0; i < 1000; ++i)
    {
        std::cout<<"creat des"<<std::endl;
        cv::Mat image = cv::imread(train_image_names[i]);
        train_images.push_back(image);
        auto fm = fe->Compute(image);
        database.add(fm.descriptors);
    }
    for(const auto& image_name: query_image_names)
    {
        cv::Mat image = cv::imread(image_name);
        auto fm = fe->Compute(image);
        DBoW3::QueryResults ret;
        database.query(fm.descriptors, ret, 5);
        cv::imshow("query", image);
        for(const auto& r: ret)
        {
            cv::imshow("train", train_images[r.Id]);
            cv::waitKey(0);
        }
    }

    return 0;

}