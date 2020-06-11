//
// Created by chenghe on 5/12/20.
//
#include <dense_mapping/densifier.h>
namespace SuperVIO::DenseMapping
{
    /////////////////////////////////////////////////////////////////////////////////////////
    cv::Mat Densifier::
    Evaluate(const cv::Mat& image, const std::vector<cv::Point2f>& points,
             const std::vector<double>& depths)
    {
        //! creat mesh
        auto triangles = DelaunayTriangulate(image.size(), points, depths);
        //! interpolate mesh
        auto pair = InterpolateMesh(image.size(), triangles);
        //! optimize depth map
        auto fine_depth_map = OptimizeDepthMap(image, pair.first, pair.second);
        //! draw functions
        auto color_image = VisualizeDepthMap(image, pair.first, fine_depth_map, triangles);

        cv::Mat confi;
        pair.second.convertTo(confi, CV_8UC1, 255, 0);
        cv::applyColorMap(confi, confi, cv::COLORMAP_JET);
        cv::hconcat(color_image, confi, color_image);

        cv::namedWindow("dense mapping", cv::WINDOW_NORMAL);
        cv::imshow("dense mapping", color_image);
        cv::waitKey(1);

        return pair.first;
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    cv::Mat Densifier::
    VisualizeDepthMap(const cv::Mat& image, const cv::Mat& raw_depth_map, const cv::Mat& fine_depth_map,
                      const std::vector<Triangle2D>& triangles)
    {
        cv::Mat color_raw_depth_map, color_fine_depth_map;

        //! draw raw depth mesh
        raw_depth_map.convertTo(color_raw_depth_map, CV_8UC1, 255.0/20.0, 0);
//        cv::normalize(color_raw_depth_map, color_raw_depth_map, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::applyColorMap(color_raw_depth_map, color_raw_depth_map, cv::COLORMAP_JET);

        //! draw fine depth image
        fine_depth_map.convertTo(color_fine_depth_map, CV_8UC1, 255.0/20.0, 0);
//        cv::normalize(color_fine_depth_map, color_fine_depth_map, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::applyColorMap(color_fine_depth_map, color_fine_depth_map, cv::COLORMAP_JET);

        //! draw mesh
        cv::Mat image_show;
        cv::cvtColor(image, image_show, CV_GRAY2BGR);
        for (const auto&triangle: triangles)
        {
            Internal::applyColorMapLine(color_fine_depth_map, triangle.b, triangle.a, &image_show);

            Internal::applyColorMapLine(color_fine_depth_map, triangle.b, triangle.c, &image_show);

            Internal::applyColorMapLine(color_fine_depth_map, triangle.c, triangle.a, &image_show);
        }

        cv::Mat output;
        cv::hconcat(color_raw_depth_map, color_fine_depth_map, output);
        cv::hconcat(output, image_show, output);

        return output;
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    std::vector<Triangle2D> Densifier::
    DelaunayTriangulate(const cv::Size& image_size, const std::vector<cv::Point2f>& points,
                        const std::vector<double>& depths)
    {
        ROS_ASSERT(points.size() == depths.size());
        ROS_ASSERT(points.size() >= 10);
        auto pair = Delaunay::Triangulate(points);
        auto triangles = MeshRegularizer::Evaluate(points, depths, pair.first, pair.second);

        return triangles;
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    struct Edge {
        static const int stepXSize = 4;
        static const int stepYSize = 1;

        // __m128 is the SSE 128-bit packed float type (4 floats).
        __m128 oneStepX;
        __m128 oneStepY;

        __m128 init(const cv::Point& v0, const cv::Point& v1,
                    const cv::Point& origin) {
            // Edge setup
            float A = v1.y - v0.y;
            float B = v0.x - v1.x;
            float C = v1.x*v0.y - v0.x*v1.y;

            // Step deltas
            // __m128i y = _mm_set1_ps(x) sets y[0..3] = x.
            oneStepX = _mm_set1_ps(A*stepXSize);
            oneStepY = _mm_set1_ps(B*stepYSize);

            // x/y values for initial pixel block
            // NOTE: Set operations have arguments in reverse order!
            // __m128 y = _mm_set_epi32(x3, x2, x1, x0) sets y0 = x0, etc.
            __m128 x = _mm_set_ps(origin.x + 3, origin.x + 2, origin.x + 1, origin.x);
            __m128 y = _mm_set1_ps(origin.y);

            // Edge function values at origin
            // A*x + B*y + C.
            __m128 A4 = _mm_set1_ps(A);
            __m128 B4 = _mm_set1_ps(B);
            __m128 C4 = _mm_set1_ps(C);

            return _mm_add_ps(_mm_add_ps(_mm_mul_ps(A4, x), _mm_mul_ps(B4, y)), C4);
        }
    };
    inline int min3(float x, float y, float z)
    {
        return std::round(x < y ? (x < z ? x : z) : (y < z ? y : z));
    }

    inline int max3(float x, float y, float z)
    {
        return std::round(x > y ? (x > z ? x : z) : (y > z ? y : z));
    }
    double Norm(const cv::Point2f& x)
    {
        return std::sqrt(x.x * x.x + x.y * x.y);
    }
    double MinAngle(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& c)
    {
        const cv::Point2f AB = a-b;
        const cv::Point2f AC = a-c;
        const cv::Point2f BC = b-c;
        double angle_c = std::acos(AC.dot(BC) / (Norm(AC) * Norm(BC))) * 180 / M_PI;
        double angle_a = std::acos((-AB).dot(-AC) / (Norm(AB) * Norm(AC))) * 180 / M_PI;
        double angle_b = std::acos(AB.dot(-BC) / (Norm(AB) * Norm(BC))) * 180 / M_PI;
        double min_angle = std::min(std::min(angle_a, angle_b), angle_c);

        return min_angle;
    }

    double Area(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& c)
    {
        double area = std::abs(0.5f * (a.x * b.y + b.x * c.y + c.x * a.y -
                                       c.x * b.y - b.x * a.y - a.x * c.y));
        return area;

    }

    /////////////////////////////////////////////////////////////////////////////////////////
    std::pair<cv::Mat, cv::Mat> Densifier::
    InterpolateMesh(const cv::Size& image_size, const std::vector<Triangle2D>& triangles)
    {
        cv::Mat depth_map = cv::Mat::zeros(image_size, CV_32FC1);
        cv::Mat angle_map = cv::Mat::zeros(image_size, CV_32FC1);
        cv::Mat area_map  = cv::Mat::zeros(image_size, CV_32FC1);
        float max_area = (image_size.width / 16.0f) * (image_size.height / 16.0f);
        for(const auto& triangle: triangles)
        {
            const auto& p1 = triangle.a;
            const auto& p2 = triangle.b;
            const auto& p3 = triangle.c;
            const auto& v1 = triangle.depth_a;
            const auto& v2 = triangle.depth_b;
            const auto& v3 = triangle.depth_c;
            float angle = MinAngle(p1, p2, p3);
            float angle_score = angle/60.0f;
            angle_score = angle > 20 ? angle / 60 : 0.2f;
            float area  = Area(p1, p2, p3);
            float area_score = (max_area - area) / max_area;
//            area_score = area_score > 0.2f ? area_score : 0.2f;
            area_score = 0.0f;
            int xmin = min3(p1.x, p2.x, p3.x);
            int ymin = min3(p1.y, p2.y, p3.y);
            int xmax = max3(p1.x, p2.x, p3.x);
            int ymax = max3(p1.y, p2.y, p3.y);

            cv::Point p(xmin, ymin);
            Edge e12, e23, e31;

            // __m128 is the SSE 128-bit packed float type (4 floats).
            __m128 w1_row = e23.init(p2, p3, p);
            __m128 w2_row = e31.init(p3, p1, p);
            __m128 w3_row = e12.init(p1, p2, p);

            // Values as 4 packed floats.
            __m128 v14 = _mm_set1_ps(v1);
            __m128 v24 = _mm_set1_ps(v2);
            __m128 v34 = _mm_set1_ps(v3);

            // Rasterize
            for (p.y = ymin; p.y <= ymax; p.y += Edge::stepYSize) {
                // Determine barycentric coordinates
                __m128 w1 = w1_row;
                __m128 w2 = w2_row;
                __m128 w3 = w3_row;

                for (p.x = xmin; p.x <= xmax; p.x += Edge::stepXSize) {
                    if(p.x < 0 || p.y < 0 || p.x >= image_size.width || p.y >= image_size.height)
                    {
                        w1 = _mm_add_ps(w1, e23.oneStepX);
                        w2 = _mm_add_ps(w2, e31.oneStepX);
                        w3 = _mm_add_ps(w3, e12.oneStepX);
                        continue;
                    }
                    // If p is on or inside all edges, render pixel.
                    __m128 zero = _mm_set1_ps(0.0f);

                    // (w1 >= 0) && (w2 >= 0) && (w3 >= 0)
                    // mask tells whether we should set the pixel.
                    __m128 mask = _mm_and_ps(_mm_cmpge_ps(w1, zero),
                                             _mm_and_ps(_mm_cmpge_ps(w2, zero),
                                                        _mm_cmpge_ps(w3, zero)));

                    // w1 + w2 + w3
                    __m128 norm = _mm_add_ps(w1, _mm_add_ps(w2, w3));

                    // v1*w1 + v2*w2 + v3*w3 / norm
                    __m128 vals = _mm_div_ps(_mm_add_ps(_mm_mul_ps(v14, w1),
                                                        _mm_add_ps(_mm_mul_ps(v24, w2),
                                                                   _mm_mul_ps(v34, w3))), norm);
                    // Grab original data.  We need to use different store/load functions if
                    // the address is not aligned to 16-bytes.
                    uint32_t addr = sizeof(float)*(p.y* depth_map.cols + p.x);
                    if (addr % 16 == 0) {
                        float* img_ptr = reinterpret_cast<float*>(&(depth_map.data[addr]));
                        float* area_ptr = reinterpret_cast<float*>(&(area_map.data[addr]));
                        float* angle_ptr = reinterpret_cast<float*>(&(angle_map.data[addr]));
                        __m128 angle_s = _mm_set1_ps(angle_score);
                        __m128 area_s  = _mm_set1_ps(area_score);
                        __m128 data = _mm_load_ps(img_ptr);
                        __m128 area_data = _mm_load_ps(area_ptr);
                        __m128 angle_data = _mm_load_ps(angle_ptr);

                        // Set values using mask.
                        // If mask is true, use vals, otherwise use data.
                        __m128 res = _mm_or_ps(_mm_and_ps(mask, vals), _mm_andnot_ps(mask, data));
                        __m128 res_angle = _mm_or_ps(_mm_and_ps(mask, angle_s), _mm_andnot_ps(mask, angle_data));
                        __m128 res_area  = _mm_or_ps(_mm_and_ps(mask, area_s), _mm_andnot_ps(mask, area_data));
                        _mm_store_ps(img_ptr, res);
                        _mm_store_ps(angle_ptr, res_angle);
                        _mm_store_ps(area_ptr, res_area);
                    } else {
                        // Address is not 16-byte aligned. Need to use special functions to load/store.
                        float* img_ptr = reinterpret_cast<float*>(&(depth_map.data[addr]));
                        float* area_ptr = reinterpret_cast<float*>(&(area_map.data[addr]));
                        float* angle_ptr = reinterpret_cast<float*>(&(angle_map.data[addr]));
                        __m128 angle_s = _mm_set1_ps(angle_score);
                        __m128 area_s  = _mm_set1_ps(area_score);
                        __m128 data = _mm_loadu_ps(img_ptr);
                        __m128 area_data = _mm_loadu_ps(area_ptr);
                        __m128 angle_data = _mm_loadu_ps(angle_ptr);

                        // Set values using mask.
                        // If mask is true, use vals, otherwise use data.
                        __m128 res = _mm_or_ps(_mm_and_ps(mask, vals), _mm_andnot_ps(mask, data));
                        __m128 res_angle = _mm_or_ps(_mm_and_ps(mask, angle_s), _mm_andnot_ps(mask, angle_data));
                        __m128 res_area  = _mm_or_ps(_mm_and_ps(mask, area_s), _mm_andnot_ps(mask, area_data));
                        _mm_storeu_ps(img_ptr, res);
                        _mm_storeu_ps(angle_ptr, res_angle);
                        _mm_storeu_ps(area_ptr, res_area);
                    }

                    // One step to the right.
                    w1 = _mm_add_ps(w1, e23.oneStepX);
                    w2 = _mm_add_ps(w2, e31.oneStepX);
                    w3 = _mm_add_ps(w3, e12.oneStepX);
                }

                // Row step.
                w1_row = _mm_add_ps(w1_row, e23.oneStepY);
                w2_row = _mm_add_ps(w2_row, e31.oneStepY);
                w3_row = _mm_add_ps(w3_row, e12.oneStepY);
            }
        }
        cv::normalize(area_map, area_map, 0, 1.0, cv::NORM_MINMAX, CV_32FC1);
        cv::normalize(angle_map, angle_map, 0, 1.0, cv::NORM_MINMAX, CV_32FC1);
        cv::Mat confidence_map = area_map + angle_map;
//        cv::normalize(confidence_map, confidence_map, 0, 1.0, cv::NORM_MINMAX, CV_32FC1);

        return std::make_pair(depth_map, confidence_map);
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    cv::Mat Densifier::
    OptimizeDepthMap(const cv::Mat& image, const cv::Mat& depth_map, const cv::Mat& confidence_map)
    {
        ROS_ASSERT(image.type() == CV_8UC1 || image.type() == CV_8UC3);
        FastBilateralSolver fbs(image);
        cv::Mat optimized_depth_map = fbs.Filter(depth_map, confidence_map);

        return optimized_depth_map;
    }


}//end of SuperVIO::DenseMapping
