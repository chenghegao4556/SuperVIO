//
// Created by chenghe on 5/12/20.
//

#ifndef SUPER_VIO_IMAGE_RETRIEVAL_H
#define SUPER_VIO_IMAGE_RETRIEVAL_H
#include <vision/feature_extractor.h>
#include <DBoW3/DBoW3.h>
#include <imu/imu_states_measurements.h>
namespace SuperVIO::LoopClosure
{
    class ImageRetrieval
    {
    public:
        class Database
        {
        public:
            typedef IMU::StateKey  FrameId;
            typedef DBoW3::EntryId EntryId;
            typedef size_t         SequenceId;

            class QueryResult
            {
            public:
                explicit QueryResult(bool _success = false);
                bool success;
                std::set<FrameId> train_frame_ids;
            };

            explicit Database(const std::string& vocabulary_path);

            [[nodiscard]] QueryResult
            Query(const cv::Mat& query_descriptors, SequenceId query_sequence_id, size_t max_num) const;

            bool Add(const cv::Mat& query_descriptors, SequenceId sequence_id, FrameId frame_id);
        private:

            std::map<EntryId, FrameId> entry_frame_map_;
            std::map<FrameId, SequenceId> frame_sequence_map_;
            DBoW3::Database bow_database_;
        };//end of DataBase

        static DBoW3::Vocabulary
        TrainVocabulary(const std::string& image_folder,
                        const std::string& weight_path,
                        const std::string& vocabulary_save_path,
                        size_t num_image);
    };
}//end of SuperVIO::LoopClosure

#endif //SUPER_VIO_IMAGE_RETRIEVAL_H
