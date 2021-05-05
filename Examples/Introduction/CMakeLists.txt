# Target name
set(target introduction)

# Define root directory
set(root_dir ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Includes
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Sources
file(GLOB header_dir
    ${root_dir}/Includes
)

file(GLOB_RECURSE headers
    ${header_dir}/*.hpp
)

file(GLOB_RECURSE sources
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
)

foreach(SESSION ${sources})
    get_filename_component(FILE_NAME ${SESSION} NAME_WE)
	add_executable(${FILE_NAME} ${SESSION})
	# Project options
    set_target_properties(${FILE_NAME}
        PROPERTIES
        ${DEFAULT_PROJECT_OPTIONS}
        VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:Sources>
        CUDA_SEPARABLE_COMPILATION ON
    )
    
    # Compile options
    target_compile_options(${FILE_NAME}
        PRIVATE
        PUBLIC
        ${DEFAULT_COMPILE_OPTIONS}
        INTERFACE
    )
    
    # Compile definitions
    target_compile_definitions(${FILE_NAME}
        PRIVATE
        RESOURCES_DIR="${RESOURCES_DIR}"
    )
    
    target_link_libraries(${FILE_NAME}
        PRIVATE
        PUBLIC
        ${DEFAULT_LINKER_OPTIONS}
    	${DEFAULT_LIBRARIES}
        CUDA::thrust
        INTERFACE
    )
    
    if(APPLE)
      # We need to add the path to the driver (libcuda.dylib) as an rpath, 
      # so that the static cuda runtime can find it at runtime.
      set_property(${FILE_NAME}
                   PROPERTY
                   BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    endif()
endforeach(PROBLEM)