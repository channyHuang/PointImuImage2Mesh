FILE(GLOB LIBRARY_FILES_C "*.cpp")
FILE(GLOB LIBRARY_FILES_H "*.h" "*.inl")

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
project(imgui)

include_directories(imgui
	${GLFW3_INCLUDE_DIR}
	)
	
add_library(imgui
${LIBRARY_FILES_C} 
${LIBRARY_FILES_H}
)

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
TARGET_LINK_LIBRARIES(imgui 
	glfw
	${CMAKE_DL_LIBS}
)
else()
TARGET_LINK_LIBRARIES(imgui 
	glfw3
)
endif()