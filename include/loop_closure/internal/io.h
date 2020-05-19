//
// Created by chenghe on 5/18/20.
//

#ifndef SUPER_VIO_IO_H
#define SUPER_VIO_IO_H
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <ros/ros.h>
#include <ros/console.h>
#include <boost/algorithm/string.hpp>
namespace SuperVIO::LoopClosure::Internal
{
    bool GetImageNames(const std::string& image_folder,
                       std::vector<std::string>& file_names)
    {
        if (image_folder.empty())
        {
            return false;
        }
        struct stat s;
        stat(image_folder.c_str(), &s);
        if (!S_ISDIR(s.st_mode))
        {
            ROS_ERROR_STREAM(" "<<image_folder<<" IS NOT A FOLDER.");
            return false;
        }
        DIR* open_dir = opendir(image_folder.c_str());
        if (nullptr == open_dir)
        {
            ROS_ERROR_STREAM("CANNOT OPEN FOLDER: "<<image_folder<<".");
            return false;
        }

        dirent* p = nullptr;
        while( (p = readdir(open_dir)) != nullptr)
        {
            if (p->d_name[0] != '.')
            {
                std::string image_name(p->d_name);
                std::vector<std::string> temp;
                boost::split(temp, image_name, boost::is_any_of("."));
                if(temp.back() == std::string("png") || temp.back() == std::string("jpg"))
                {
                    file_names.push_back(image_folder + std::string("/") + image_name);
                }
                else
                {
                    ROS_WARN_STREAM("UNSUPPORT IMAGE FORMAT!");
                }
            }
        }

        closedir(open_dir);
        return true;
    }
}//end of SuperVIO::LoopClosure::Internal

#endif //SUPER_VIO_IO_H
