//
// Created by chenghe on 5/12/20.
//
#include <loop_closure/image_retrieval.h>
#include <loop_closure/internal/io.h>
namespace SuperVIO::LoopClosure
{
    /////////////////////////////////////////////////////////////////////////////////////////
    ImageRetrieval::Database::QueryResult::
    QueryResult(bool _success):
        success(_success)
    {
        
    }
    
    /////////////////////////////////////////////////////////////////////////////////////////
    ImageRetrieval::Database::
    Database(const std::string& vocabulary_path)
    {
        DBoW3::Vocabulary vocabulary(vocabulary_path);
        bow_database_.setVocabulary(vocabulary, false, 0);
    }

    /////////////////////////////////////////////////////////////////////////////////////////     
    bool ImageRetrieval::Database::
    Add(const cv::Mat& query_descriptors, SequenceId sequence_id, FrameId frame_id)
    {
        auto entry_id = bow_database_.add(query_descriptors);
        ROS_ASSERT(entry_frame_map_.find(entry_id) == entry_frame_map_.end());
        entry_frame_map_.insert(std::make_pair(entry_id, frame_id));
        ROS_ASSERT(frame_sequence_map_.find(entry_id) == frame_sequence_map_.end());
        frame_sequence_map_.insert(std::make_pair(frame_id, sequence_id));

        return true;
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    ImageRetrieval::Database::QueryResult ImageRetrieval::Database::
    Query(const cv::Mat& query_descriptors, SequenceId query_sequence_id, size_t max_num) const
    {
        DBoW3::QueryResults ret;
        bow_database_.query(query_descriptors, ret, max_num);
        std::map<SequenceId, size_t> sequence_count;
        std::map<FrameId, SequenceId> sequence_map;
        //! check best candidate's score
        if(ret.front().Score < 0.05)
        {
            return QueryResult();
        }

        for(auto& iter: ret)
        {
            auto train_frame_id = entry_frame_map_.at(iter.Id);
            auto train_sequence_id  = frame_sequence_map_.at(train_frame_id);

            if(query_sequence_id == train_sequence_id || iter.Score < 0.015)
            {
                continue;
            }

            if(sequence_count.find(train_sequence_id) == sequence_count.end())
            {
                sequence_count.insert(std::make_pair(train_sequence_id, 1));
            }
            else
            {
                sequence_count.at(train_sequence_id) += 1;
            }

            sequence_map.insert(std::make_pair(train_frame_id, train_sequence_id));
        }
        std::set<SequenceId> keep_sequence_ids;
        for(const auto& count: sequence_count)
        {
            if(count.second >= 2)
            {
                keep_sequence_ids.insert(count.first);
            }
        }
        if(keep_sequence_ids.empty())
        {
            return QueryResult();
        }
        QueryResult result(true);
        for(const auto& f_s: sequence_map)
        {
            if(keep_sequence_ids.find(f_s.second) != keep_sequence_ids.end())
            {
                result.train_frame_ids.insert(f_s.first);
            }
        }

        return result;
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    DBoW3::Vocabulary ImageRetrieval::
    TrainVocabulary(const std::string& image_folder,
                    const std::string& weight_path,
                    const std::string& vocabulary_save_path,
                    size_t num_image)
    {
        std::vector<std::string> image_names;
        DBoW3::Vocabulary vocabulary(10, 5, DBoW3::TF_IDF, DBoW3::L2_NORM);
        if(Internal::GetImageNames(image_folder, image_names))
        {
            std::vector<cv::Mat> descriptors;
            Vision::FeatureExtractor::Parameters params;
            params.inference_resolution = cv::Size(320, 240);
            auto feature_extractor = Vision::FeatureExtractor::Creat(params, weight_path);
            for(const auto& image_name: image_names)
            {
                ROS_INFO_STREAM("creat descriptor .......");
                cv::Mat image = cv::imread(image_name, 0);
                auto fm = feature_extractor->Compute(image);
                descriptors.push_back(fm.descriptors);
                if(descriptors.size() >= num_image)
                {
                    break;
                }
            }

            if(descriptors.size() > 100)
            {
                ROS_INFO_STREAM("START TRAINING VOCABULARY .......");
                vocabulary.create(descriptors);
                ROS_INFO_STREAM("FINISH TRAINING VOCABULARY");
                vocabulary.save(vocabulary_save_path);
            }
            else
            {
                ROS_ERROR_STREAM("NEED AT LEAST 100 IMAGES!");
            }
        }

        return vocabulary;
    }
}//end of SuperVIO::LoopClosure