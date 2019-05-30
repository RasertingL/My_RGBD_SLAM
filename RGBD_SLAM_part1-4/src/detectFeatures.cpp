#include <iostream>
#include "slamBase.h"
//下面是opencv特征检测模块
#include <opencv2/features2d/features2d.hpp>
// #include <opencv2/nonfree/nonfree.hpp> //我们这里使用ORB特征点,如果要用SIFT,就用这个头文件
#include <opencv2/calib3d/calib3d.hpp>


using namespace std ;
using namespace cv ;

void find_feature_matches(
    const Mat & img_1, const Mat & img_2, 
    std::vector<KeyPoint>& KeyPoint_1,
    std::vector<KeyPoint>& KeyPoint_2,
    std::vector<DMatch>& matches
) ;

void pose_estimation_3d2d(
    std::vector<KeyPoint>& keypoints_1,
    std::vector<KeyPoint>& keypoints_2,
    std::vector<DMatch>& matches,
    Mat& depth,
    Mat& R,
    Mat& t
) ;




int main(){
    Mat img_1 = cv::imread( "./data/rgb1.png");
    Mat img_2 = cv::imread( "./data/rgb2.png");
    Mat depth1 = cv::imread( "./data/depth1.png", -1);
    Mat depth2 = cv::imread( "./data/depth2.png", -1);

    vector<KeyPoint> keypoint_1, keypoint_2 ;
    vector<DMatch> matches ;
    find_feature_matches(img_1, img_2, keypoint_1, keypoint_2, matches) ;
    cout << "一共得到了" << matches.size() << "组匹配点" << endl ;

    //对于两张图的运动估计,关键函数:cv::solvePnP()
    Mat R, t ;
    pose_estimation_3d2d(keypoint_1, keypoint_2, matches, depth1, R, t) ;
    cout << "旋转R:" << R << endl ;
    cout << "平移t" << t << endl ;

}


void find_feature_matches ( const Mat& img_1, const Mat& img_2,
                            std::vector<KeyPoint>& keypoints_1,
                            std::vector<KeyPoint>& keypoints_2,
                            std::vector< DMatch >& matches )
{
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3 
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2 
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    //BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, match );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            matches.push_back ( match[i] );
        }
    }
}


void pose_estimation_3d2d(  std::vector<KeyPoint>& keypoints_1,
                            std::vector<KeyPoint>& keypoints_2,
                            std::vector<DMatch>& matches,
                            Mat& depth,
                            Mat& R,
                            Mat& t
                        )
{
    //相机内参
    CAMERA_INTRINSIC_PARAMETERS C ; 
    C.scale = 1000 ;
    C.cx = 325.5 ;
    C.cy = 253.5 ;
    C.fx = 518.0 ;
    C.fy = 519.0 ;

    //第一个帧的三维点
    vector<Point3f> points1 ;
    //第二个帧的图像点
    vector<Point2f> points2 ;

    depth = cv::imread( "./data/depth1.png", -1);

    for(size_t i = 0; i < matches.size(); i++){

        // query 是第一个, train 是第二个
        Point2f p = keypoints_1[matches[i].queryIdx].pt ;

        //获取d的时候要注意:x是向右的,y是向下的,所以y才是行,x是列
        ushort d = depth.ptr<ushort>(int(p.y))[int(p.x)] ;
        if(d == 0)
            continue ;

        points2.push_back(Point2f(keypoints_2[matches[i].trainIdx].pt)) ;
        /*这一块两个points1/2的赋值语句中
        分别为第i+1对特征点对（角标从0开始，i=0对应第一对特征点对）中，
        前一帧（查询图像）中的特征点索引在keypoints_1中对应的特征点坐标，
        与后一帧（训练图像）中的特征点索引在keypoints_2中对应的特征点坐标
        */

        //将(u, v, d)转化为(x, y, z)
        Point3f pt (p.x, p.y, d) ;
        Point3f pd = point2dTo3d(pt, C) ;
        points1.push_back(pd) ;
    }

    double camera_matrix_data[3][3] = {
        {C.fx, 0, C.cx}, 
        {0, C.fy, C.cy},
        {0, 0, 1}
    } ;

    //构建相机矩阵
    Mat cameraMatrix(3, 3, CV_64F, camera_matrix_data) ;

    Mat inliers ;

    //求解pnp
    solvePnP(points1, points2, cameraMatrix, Mat(), R, t, false, SOLVEPNP_EPNP) ;
    
}


