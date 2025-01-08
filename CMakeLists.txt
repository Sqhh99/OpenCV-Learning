cmake_minimum_required(VERSION 3.16)

project(Sq_OpencvTest01 VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# link_directories(${CMAKE_CURRENT_SOURCE_DIR}/bin)
file(GLOB OPENCV_LIBS ${CMAKE_CURRENT_SOURCE_DIR}/lib/libopencv_*.a)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)

set(PROJECT_SOURCES
        main.cpp
        mainwindow.cpp
        mainwindow.h
        mainwindow.ui
)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(Sq_OpencvTest01
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
        Resources.qrc
        opencvtools.h opencvtools.cpp

    )
else()
    if(ANDROID)
        add_library(Sq_OpencvTest01 SHARED
            ${PROJECT_SOURCES}
        )
# Define properties for Android with Qt 5 after find_package() calls as:
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
    else()
        add_executable(Sq_OpencvTest01
            ${PROJECT_SOURCES}
        )
    endif()
endif()

target_link_libraries(Sq_OpencvTest01 PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)
target_link_libraries(Sq_OpencvTest01 PRIVATE ${OPENCV_LIBS})
# target_link_libraries(Sq_OpencvTest01 PRIVATE
#     opencv_core490
#     opencv_imgproc490
#     opencv_highgui490
#     opencv_video490
#     opencv_videoio490
#     opencv_dnn490
# )


if(${QT_VERSION} VERSION_LESS 6.1.0)
  set(BUNDLE_ID_OPTION MACOSX_BUNDLE_GUI_IDENTIFIER com.example.Sq_OpencvTest01)
endif()
set_target_properties(Sq_OpencvTest01 PROPERTIES
    ${BUNDLE_ID_OPTION}
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

include(GNUInstallDirs)
install(TARGETS Sq_OpencvTest01
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(Sq_OpencvTest01)
endif()
