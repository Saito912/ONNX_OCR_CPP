cmake_minimum_required(VERSION 3.15)

project(MyOnnxApp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime)

include_directories(${ONNXRUNTIME_DIR}/include)

# 查找 ONNX Runtime 的库文件 (onnxruntime.lib)
find_library(ONNXRUNTIME_LIB
        NAMES onnxruntime
        PATHS ${ONNXRUNTIME_DIR}/lib
        REQUIRED
)


set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv)

# 查找 OpenCV 包，并指定版本和需要的模块
find_package(OpenCV 4.10 REQUIRED
        COMPONENTS core highgui imgcodecs imgproc
)

if(OpenCV_FOUND)
    message(STATUS "OpenCV found at: ${OpenCV_DIR}")
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()



include_directories(${CMAKE_SOURCE_DIR}/src)

file(GLOB SRC_FILES "src/*.cpp")


add_executable(MyOnnxApp main.cpp ${SRC_FILES})

target_link_libraries(MyOnnxApp PRIVATE
        ${ONNXRUNTIME_LIB}
        ${OpenCV_LIBS}
)

message(STATUS "Configuring post-build steps to copy DLLs...")

add_custom_command(TARGET MyOnnxApp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:MyOnnxApp>
        COMMENT "Copying onnxruntime.dll to output directory"
)

set(OPENCV_BIN_DIR ${OpenCV_DIR}/x64/vc16/bin)
file(GLOB OPENCV_DLLS "${OPENCV_BIN_DIR}/*.dll")

if(OPENCV_DLLS)
    add_custom_command(TARGET MyOnnxApp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${OPENCV_DLLS}
            $<TARGET_FILE_DIR:MyOnnxApp>
            COMMENT "Copying OpenCV DLLs to output directory"
    )
else()
    message(WARNING "Could not find OpenCV DLLs at ${OPENCV_BIN_DIR}. You may need to copy them manually to run the executable.")
endif()