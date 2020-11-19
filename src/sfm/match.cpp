//
// Created by chenghe on 3/23/20.
//
#include <sfm/match.h>

namespace SuperVIO::SFM
{
    /////////////////////////////////////////////////////////////////////////////////////////////
    Match::
    Match(size_t _point_i_id,
          size_t _point_j_id,
          size_t _track_id):
            point_i_id(_point_i_id),
            point_j_id(_point_j_id),
            track_id(_track_id)
    {

    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    Pose::
    Pose(bool _success,
         const Quaternion& _rotation,
         Vector3 _position):
            success(_success),
            rotation(_rotation),
            position(std::move(_position))
    {

    }


}//end of SuperVIO
