//
// Created by chenghe on 5/12/20.
//
#include <dense_mapping/densifier.h>
namespace SuperVIO::DenseMapping
{
    /////////////////////////////////////////////////////////////////////////////////////////
    Densifier::Triangle2D::
    Triangle2D(const cv::Point2f& _a, const cv::Point2f& _b, const cv::Point2f& _c,
               double _d_a, double _d_b, double _d_c):
                a(_a),
                b(_b),
                c(_c),
                depth_a(_d_a),
                depth_b(_d_b),
                depth_c(_d_c)
    {

    }

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
        cv::Mat color_depth_map, color_fine_depth_map;
        pair.first.convertTo(color_depth_map, CV_8UC1, 255.0/20.0, 0);
        fine_depth_map.convertTo(color_fine_depth_map, CV_8UC1, 255.0/20.0, 0);
        cv::normalize(color_depth_map, color_depth_map, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::normalize(color_fine_depth_map, color_fine_depth_map, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::applyColorMap(color_depth_map, color_depth_map, cv::COLORMAP_JET);
        cv::applyColorMap(color_fine_depth_map, color_fine_depth_map, cv::COLORMAP_JET);
        cv::Mat image_show;
        cv::cvtColor(image, image_show, CV_GRAY2BGR);
        cv::Mat out;
        cv::hconcat(color_depth_map, color_fine_depth_map, out);
        cv::hconcat(out, image_show, out);
        cv::namedWindow("dense mapping", cv::WINDOW_NORMAL);
        cv::imshow("dense mapping", out);
        cv::waitKey(1);

        return pair.first;
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    double get_clockwise_angle(const std::pair<float, float>& p)
    {
        double angle = -std::atan2(p.first,-p.second);
        return angle;
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    bool compare_points(const std::pair<float, float>& a, const std::pair<float, float>& b)
    {
        return (get_clockwise_angle(a) >= get_clockwise_angle(b));
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    std::vector<Densifier::Triangle2D> Densifier::
    DelaunayTriangulate(const cv::Size& image_size, const std::vector<cv::Point2f>& points,
                        const std::vector<double>& depths)
    {
        ROS_ASSERT(points.size() == depths.size());
        ROS_ASSERT(points.size() >= 10);
        std::map<std::pair<float, float>, double> pd_map;
        for(size_t i = 0; i < points.size(); ++i)
        {
            pd_map.insert(std::make_pair(std::make_pair(points[i].x, points[i].y), depths[i]));
        }
        auto sub = cv::Subdiv2D(cv::Rect(0, 0, image_size.width, image_size.height));
        sub.insert(points);
        std::vector<cv::Vec6f> cv_triangles;
        sub.getTriangleList(cv_triangles);
        std::vector<Triangle2D> result;
        for(const auto& tri: cv_triangles)
        {
            std::vector<std::pair<float, float>> ps;
            ps.emplace_back(tri[0], tri[1]);
            ps.emplace_back(tri[2], tri[3]);
            ps.emplace_back(tri[4], tri[5]);
            std::sort(ps.begin(), ps.end(), compare_points);
            if(pd_map.count(ps[0]) && pd_map.count(ps[1]) && pd_map.count(ps[2]))
            {
                auto d_a = pd_map.find(ps[0])->second;
                auto d_b = pd_map.find(ps[1])->second;
                auto d_c = pd_map.find(ps[2])->second;
                result.emplace_back(cv::Point2f(ps[0].first, ps[0].second), cv::Point2f(ps[1].first, ps[1].second),
                                    cv::Point2f(ps[2].first, ps[2].second), d_a, d_b, d_c);
            }
        }

        return result;
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

    /////////////////////////////////////////////////////////////////////////////////////////
    std::pair<cv::Mat, cv::Mat> Densifier::
    InterpolateMesh(const cv::Size& image_size, const std::vector<Triangle2D>& triangles)
    {
        cv::Mat depth_map = cv::Mat::zeros(image_size, CV_32FC1);
        cv::Mat confidence_map = cv::Mat::zeros(image_size, CV_32FC1);
        for(const auto& triangle: triangles)
        {
            const auto& p1 = triangle.a;
            const auto& p2 = triangle.b;
            const auto& p3 = triangle.c;
            const auto& v1 = triangle.depth_a;
            const auto& v2 = triangle.depth_b;
            const auto& v3 = triangle.depth_c;
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
                        float* confidence_ptr = reinterpret_cast<float*>(&(confidence_map.data[addr]));
                        __m128 one = _mm_set1_ps(1.0f);
                        __m128 data = _mm_load_ps(img_ptr);
                        __m128 confidence_data = _mm_load_ps(confidence_ptr);

                        // Set values using mask.
                        // If mask is true, use vals, otherwise use data.
                        __m128 res = _mm_or_ps(_mm_and_ps(mask, vals), _mm_andnot_ps(mask, data));
                        __m128 res_one = _mm_or_ps(_mm_and_ps(mask, one), _mm_andnot_ps(mask, confidence_data));
                        _mm_store_ps(img_ptr, res);
                        _mm_store_ps(confidence_ptr, res_one);
                    } else {
                        // Address is not 16-byte aligned. Need to use special functions to load/store.
                        float* img_ptr = reinterpret_cast<float*>(&(depth_map.data[addr]));
                        float* confidence_ptr = reinterpret_cast<float*>(&(confidence_map.data[addr]));
                        __m128 data = _mm_loadu_ps(img_ptr);
                        __m128 one = _mm_set1_ps(1.0f);
                        __m128 confidence_data = _mm_loadu_ps(confidence_ptr);

                        // Set values using mask.
                        // If mask is true, use vals, otherwise use data.
                        __m128 res = _mm_or_ps(_mm_and_ps(mask, vals), _mm_andnot_ps(mask, data));
                        __m128 res_one = _mm_or_ps(_mm_and_ps(mask, one), _mm_andnot_ps(mask, confidence_data));
                        _mm_storeu_ps(img_ptr, res);
                        _mm_storeu_ps(confidence_ptr, res_one);
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
