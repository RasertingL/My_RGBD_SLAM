#include "slamBase.h"
#include <opencv2/core/eigen.hpp>

#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>

// Eigen !
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std ;
using namespace cv ;

int main(){

    //相机内参
    CAMERA_INTRINSIC_PARAMETERS C ; 
    C.scale = 1000 ;
    C.cx = 325.5 ;
    C.cy = 253.5 ;
    C.fx = 518.0 ;
    C.fy = 519.0 ;

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

    //使用罗德里格变换(Rodrigues)将旋转向量转化为旋转矩阵
    Mat R1 ;
    Rodrigues(R, R1) ;
    Eigen::Matrix3d r ;
    //mat旋转矩阵->eigen旋转矩阵
    cv::cv2eigen(R1, r) ;

    //将旋转矩阵与平移向量转成变换矩阵
    //定义变换矩阵
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity() ;
    //eigen旋转矩阵->eigen轴角
    Eigen::AngleAxisd angle(r) ;

    //Eigen::Translation<double,3> trans(t.at<double>(0,0), t.at<double>(0,1), t.at<double>(0,2));
    //eigen轴角->eigen变换矩阵旋转部分
    T = angle ;
    //平移部分直接按位置赋值过去
    T(0, 3) = t.at<double>(0, 0) ;
    T(1, 3) = t.at<double>(0, 1) ;
    T(2, 3) = t.at<double>(0, 2) ;
    

    //点云转换
    PointCloud::Ptr cloud1 = image2PointCloud(img_1, depth1, C) ;
    PointCloud::Ptr cloud2 = image2PointCloud(img_2, depth2, C) ;

    //合并点云
    PointCloud::Ptr output(new PointCloud()) ;
    //变换矩阵 * 相机坐标,得到世界坐标(相机坐标根据外参,也就是变换矩阵,变到世界坐标)
    pcl::transformPointCloud( *cloud1, *output, T.matrix() );
    //上面T也要用.matrix()方法获得矩阵类型，因为T是Eigen::Isometry3d类型，而这里需要的是Eigen::Matrix类型，其实并不对等。需要用.matrix()方法获得矩阵
    *output += *cloud2 ;//+=操作也是对应点云，点云指针不可以

    //保存
    cout<<"点云共有"<<output->size()<<"个点."<<endl;
    pcl::io::savePCDFileBinary("map.pcd", *output );


}