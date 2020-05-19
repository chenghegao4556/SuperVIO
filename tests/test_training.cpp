//
// Created by chenghe on 5/18/20.
//
#include <loop_closure/image_retrieval.h>
#include <loop_closure/internal/io.h>
using namespace SuperVIO::LoopClosure;

int main()
{
    ImageRetrieval::TrainVocabulary("/media/chenghe/ChengheGao/datasets/coco/train2017",
            "/home/chenghe/catkin_ws/src/SuperVIO/data/superpoint.pt", "vocabulary.yml.gz", 5000);
}
