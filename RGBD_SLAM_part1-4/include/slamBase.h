//说明:将之前的2D转3D坐标的代码制作为一个叫做slamBase的库

# pragma once

#include<iostream>
#include<string>
#include<vector>
#include<fstream>
#include <opencv2/features2d/features2d.hpp>
// #include <opencv2/nonfree/nonfree.hpp> //我们这里使用ORB特征点,如果要用SIFT,就用这个头文件
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

//PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <opencv2/core/eigen.hpp>

#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>

// Eigen !
#include <Eigen/Core>
#include <Eigen/Geometry>



using namespace std ;
using namespace cv ;

//定义点云类型
typedef pcl::PointXYZRGBA PointT ;
typedef pcl::PointCloud<PointT> PointCloud ;

//相机内参结构
struct  CAMERA_INTRINSIC_PARAMETERS
{
    /* data */
    double cx, cy, fx, fy, scale ;
};

//函数接口

//image2PointCloud 将rgb转化为点云
PointCloud::Ptr image2PointCloud(Mat &rgb, Mat &depth, CAMERA_INTRINSIC_PARAMETERS &camera) ;

Point3f point2dTo3d(Point3f &point, CAMERA_INTRINSIC_PARAMETERS &camera) ;

//进行特征匹配的函数,同时提取特征关键点和描述子.
//并且返回方式为传引用.
void find_feature_matches(
    const Mat & img_1, const Mat & img_2, 
    std::vector<KeyPoint>& KeyPoint_1,
    std::vector<KeyPoint>& KeyPoint_2,
    std::vector<DMatch>& matches
) ;


//计算两个帧之间的运动.
//返回方式为传引用,旋转R,平移t
void pose_estimation_3d2d(
    std::vector<KeyPoint>& keypoints_1,
    std::vector<KeyPoint>& keypoints_2,
    std::vector<DMatch>& matches,
    Mat& depth,
    Mat& R,
    Mat& t
) ;