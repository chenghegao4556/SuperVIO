//
// Created by chenghe on 3/23/20.
//


#ifndef SUPER_VIO_MATCH_H
#define SUPER_VIO_MATCH_H

#include <vector>
#include <utility/eigen_type.h>
namespace SuperVIO::SFM
{
    class Match
    {
    public:
        Match(size_t _point_i_id,
              size_t _point_j_id,
              size_t _track_id);

        size_t point_i_id;
        size_t point_j_id;
        size_t track_id;
    };

    class Pose
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        explicit Pose(bool _success = false,
                      const Quaternion& _rotation = Quaternion::Identity(),
                      Vector3 _position = Vector3::Zero());

        bool success;
        Quaternion rotation;
        Vector3 position;
    };

    typedef std::vector<Match> Matches;
}//end of SuperVIO


#endif //SUPER_VIO_MATCH_H