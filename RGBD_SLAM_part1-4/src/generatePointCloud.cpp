#include<iostream>
#include<string>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <pcl/point_types.h> //pcl模块
#include <pcl/io/pcd_io.h> //pcl模块
#include <pcl/visualization/pcl_visualizer.h> //pcl模块


using namespace std ;
using namespace cv ;

//定义点云类型
typedef pcl::PointXYZRGBA PointT ;
typedef pcl::PointCloud<PointT> PointCloud ;

//相机内参
const double camera_factor = 1000 ;
const double camera_cx = 325.5 ;
const double camera_cy = 253.5 ;
const double camera_fx = 518.0 ;
const double camera_fy = 519.0 ;


int main(){
    //首先读取图片,包括RGB图和深度图
    //rgb图像是8UC3的彩色图像
    //depth图是16UC1的单通道图,
    Mat rgb = imread("./data/rgb.png") ;
    Mat depth = imread("./data/depth.png", -1) ;

    //点云变量,使用只能指针创建一个空的点云
    PointCloud::Ptr cloud (new PointCloud) ;

    //遍历深度图
    //这里用ptr方法,因为比at更加高效
    for(int m = 0 ; m < depth.rows ; m++){
        ushort* dptr = depth.ptr<ushort>(m) ;
        for(int n = 0 ; n < depth.cols ; n++){
            //获取深度图中的(m,n)处的点
            ushort d = dptr[n] ;

            //d可能没有值,如果没有就跳过
            if(d == 0)
                continue ;

            //反之,如果d存在,就向点云增加一个点
            PointT p ;

            //计算这个点的空间坐标
            p.z = double(d) / camera_factor ;
            p.x = (n - camera_cx) * p.z / camera_fx ;
            p.y = (m - camera_cy) * p.z / camera_fy ;

            //从rgb图像中获得颜色
            p.b = rgb.ptr<uchar>(m)[n * 3] ;
            p.g = rgb.ptr<uchar>(m)[n * 3 + 1] ;
            p.r = rgb.ptr<uchar>(m)[n * 3 + 2] ;

            //把p加入到点云
            cloud->points.push_back(p) ;
        }
    }
    //设置并保存点云
    cloud->height = 1 ;
    cloud->width = cloud->points.size() ;
    cout << "point cloud size = " << cloud->points.size() << endl ;
    cloud->is_dense = false ;
    pcl::io::savePCDFile("./pointcloud.pcd", *cloud) ;
    cout << "Points cloud saved" << endl ;
    //清除数据并退出
    cloud->points.clear() ;

    return 0 ;

    
}