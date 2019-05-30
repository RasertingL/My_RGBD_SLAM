#include "slamBase.h"
#include <iostream>
#include <sstream>

using namespace std ;
using namespace cv ;
using namespace Eigen ;

//读取一帧图片
void readFrame(int index, Mat &rgb, Mat &depth) ;

//度量运动的大小
double normofTransform(Mat &rvec, Mat &tvec) ;

int main(){
    //设置读取图片的开始和结束index ;
    int startIndex = 1 ;
    int endIndex =700 ;
    //设置是否实时显示点云
    bool visualize = true ;
    //最小内点
    int min_inliers = 5 ;
    //最大运动误差
    int max_norm = 0.3 ;

    //开始初始化配置
    //先前图片的rgb图与depth图
    Mat rgb ;
    Mat depth ;
    //新来的第二章图片的rgb与depth图
    Mat rgb2 ;
    Mat depth2 ;

    //相机内参
    CAMERA_INTRINSIC_PARAMETERS C ; 
    C.scale = 1000 ;
    C.cx = 325.5 ;
    C.cy = 253.5 ;
    C.fx = 518.0 ;
    C.fy = 519.0 ;

    int currentIndex = startIndex ; //当前索引配置
    readFrame(currentIndex, rgb, depth) ;
    //rgb = imread("./data/rgb_png/1.png") ;
    //depth = imread("./data/depth_png/1.png") ;

    //建立第一张图的点云
    PointCloud::Ptr cloud = image2PointCloud(rgb, depth, C) ;
    //保存一下第一个图的点云
    pcl::io::savePCDFile("./first_pic.pcd", *cloud) ;
    //建立点云显示器
    pcl::visualization::CloudViewer viewer("viewer") ;

    //建立vector容器对之后的关键点与匹配进行保存
    vector<KeyPoint> keypoint_1, keypoint_2 ;
    vector<DMatch> matches ;
    //旋转向量与平移向量
    Mat R, t ;

    for(currentIndex = startIndex + 1 ; currentIndex < endIndex ; currentIndex ++){
        readFrame(currentIndex, rgb2, depth2) ;
        cout << "读完图片:" << currentIndex << endl ;
        find_feature_matches(rgb, rgb2, keypoint_1, keypoint_2, matches) ;
        cout << "一共得到了" << matches.size() << "组匹配点" << endl ;

        //对于两张图的运动估计,关键函数:cv::solvePnP()
        pose_estimation_3d2d(keypoint_1, keypoint_2, matches, depth, R, t) ;   

        //对关键点与匹配点清空,防止每次累加
        keypoint_1.clear() ;
        keypoint_2.clear() ;
        matches.clear() ;

        //计算运动范围是否太大
        double norm = normofTransform(R, t) ;
        cout << "norm = " << norm << endl ;
        //这里注释掉了这个判断，因为很奇怪，如果有这个判断，会显示不出图，并且在运行到200都幅图的时候退报错然后退出。
        //if(norm >= max_norm){
        //    continue ;
        //}

        //得到变换矩阵
        Eigen::Isometry3d T = cvMat2Eigen(R, t) ;
        cout << "T = " << T.matrix() << endl ;

        //进行点云拼接
        cloud = joinPointCloud(cloud, rgb2, depth2, T, C) ;

        if(visualize)
            viewer.showCloud(cloud) ;

        rgb = rgb2.clone() ;
        depth = depth2.clone() ;
    }
    pcl::io::savePCDFile("./VO.pcd", *cloud) ;
}


void readFrame(int index, Mat &rgb, Mat &depth){
    //设置文件目录路径
    string rgb_dir = "./data/rgb_png/" ;
    string depth_dir = "./data/depth_png/" ;
    //设置文件后缀
    string rgb_Ext = ".png" ;
    string depth_Ext = ".png" ;

    // 字符串流。这个是在std中的，不是在sstream。
    // 用法上跟iostream没啥区别，只不过进入此流的数据全部被转化成字符串，而流出类型也是字符串
    stringstream ss ;
    string filename ;

    //读rgb图
    ss<<rgb_dir<<index<<rgb_Ext ; //这里可以看出，index是int类型，也可以进入流中。
    ss>>filename ; //输出之后,整个就变成了sting类型
    rgb = imread(filename) ;

    //对字符串与字符串流进行清空
    ss.clear() ;
    filename.clear() ;

    //读depth图
    ss<<depth_dir<<index<<depth_Ext ; //这里可以看出，index是int类型，也可以进入流中。
    ss>>filename ; //输出之后,整个就变成了sting类型
    depth = imread(filename, -1) ;
}

double normofTransform(Mat &rvec, Mat &tvec){
    //这里注意旋转的周期性。旋转向量的模长表示旋转角度，所以旋转角度要取自身和2PI-自身的最小值。要取到劣弧，不要优弧
    return fabs(min(cv::norm(rvec), 2*M_PI-cv::norm(rvec)))+ fabs(cv::norm(tvec));
}