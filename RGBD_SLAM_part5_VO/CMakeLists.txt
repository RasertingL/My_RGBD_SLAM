cmake_minimum_required( VERSION 2.8 )
project( rgbdslam )

# set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )  # 设定编译器

SET( CMAKE_CXX_COMPILER "g++")
SET( CMAKE_BUILD_TYPE Release  )
SEt( CMAKE_CXX_FLAGS "-O3" )

#设定可执行二进制文件的目录
SET( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin) 
	
#设定存放编译出来的库文件的目录
SET( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib) 
#并且把该目录设为连接目录
LINK_DIRECTORIES( ${PROJECT_SOURCE_DIR}/lib)
	
#设定头文件目录
INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR}/include)
	
#增加子文件夹，也就是进入源代码文件夹继续构建
ADD_SUBDIRECTORY( ${PROJECT_SOURCE_DIR}/src)