# 增加一个可执行的二进制
ADD_EXECUTABLE( main main.cpp )

# 增加PCL库的依赖
FIND_PACKAGE( PCL REQUIRED COMPONENTS common io )

list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4") # use this in Ubuntu 16.04

# 增加opencv的依赖
FIND_PACKAGE( OpenCV REQUIRED )

# 添加头文件和库文件
ADD_DEFINITIONS( ${PCL_DEFINITIONS} )
INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS}  )
LINK_LIBRARIES( ${PCL_LIBRARY_DIRS} )

ADD_EXECUTABLE( generate_pointcloud generatePointCloud.cpp )
TARGET_LINK_LIBRARIES( generate_pointcloud ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )

# 将slamBase编译成库
ADD_LIBRARY( slambase slamBase.cpp )
# 并将该库里使用的opencv和pcl库连接起来
# 但是这里注意:这里并没有将slamBase库和调用他的main函数连接,而是在下面那一块进行连接
TARGET_LINK_LIBRARIES( slambase
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )

# 下面是将特征匹配代码与各个库链接起来
# ADD_EXECUTABLE( detectFeatures detectFeatures.cpp )
# TARGET_LINK_LIBRARIES( detectFeatures 
#     slambase
#     ${OpenCV_LIBS} 
#     ${PCL_LIBRARIES} )

#将点云拼接代码和各个库链接起来
ADD_EXECUTABLE( joinPointCloud joinPointCloud.cpp)
TARGET_LINK_LIBRARIES( joinPointCloud
    slambase
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )

