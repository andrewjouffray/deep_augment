cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

link_directories(${CMAKE_SOURCE_DIR}/libs/tinymxl/)

project( maintest )

find_package(OpenMP REQUIRED)
find_package( OpenCV REQUIRED )
add_library(HyperAugment ./src/Dataset.cpp)
add_library(tinyxml ./libs/tinyxml/tinyxml.cpp ./libs/tinyxml/tinyxmlerror.cpp ./libs/tinyxml/tinyxmlparser.cpp ./libs/tinyxml/tinystr.cpp)
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( maintest ./src/main.cpp )
if(OpenMP_CXX_FOUND)
	target_link_libraries(maintest PUBLIC OpenMP::OpenMP_CXX ${OpenCV_LIBS} HyperAugment tinyxml)
endif()
