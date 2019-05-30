# 增加PCL库的依赖
# 注意 这里和前面几次的pcl引入不同,有修改,增加了对visualization 以及 filters的引入支持
FIND_PACKAGE( PCL REQUIRED COMPONENTS common io visualization filters)

list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4") # use this in Ubuntu 16.04

# 增加opencv的依赖
FIND_PACKAGE( OpenCV REQUIRED )
# INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

# 添加头文件和库文件
ADD_DEFINITIONS( ${PCL_DEFINITIONS} )
INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS}  )
LINK_LIBRARIES( ${PCL_LIBRARY_DIRS} )


# 将slamBase编译成库
ADD_LIBRARY( slambase slamBase.cpp )
# 并将该库里使用的opencv和pcl库连接起来
# 但是这里注意:这里并没有将slamBase库和调用他的main函数连接,而是在下面那一块进行连接
TARGET_LINK_LIBRARIES( slambase
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )


ADD_EXECUTABLE( visualOdometry visualOdometry.cpp)
TARGET_LINK_LIBRARIES( visualOdometry
    slambase
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )


