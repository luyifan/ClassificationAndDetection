cmake_minimum_required(VERSION 2.8.3)
set(Python_ADDITIONAL_VERSIONS 2.7 2.6 2.5)

FIND_PACKAGE(PythonInterp)
FIND_PACKAGE(PythonLibs)
FIND_PACKAGE(Boost COMPONENTS python)
find_package( OpenCV REQUIRED )
ENABLE_TESTING()
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${OPENCV_INCLUDE_DIRS})
LINK_LIBRARIES(${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ) # Deprecated but so convenient!
set(Bing "bing")
add_library(${Bing} SHARED bing.cpp)
set_target_properties(${Bing} PROPERTIES PREFIX "")
TARGET_LINK_LIBRARIES(${Bing} opencv_core opencv_imgproc opencv_video opencv_highgui ${EXTERNAL_LIBS})
