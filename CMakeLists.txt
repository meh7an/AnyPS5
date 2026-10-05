cmake_minimum_required(VERSION 3.20)
project(Relinker CXX)

set(ANYPS5_TEST_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")

function(add_test_executable target)
    add_executable(${target} ${ARGN})
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<1:${ANYPS5_TEST_OUTPUT_DIRECTORY}>")
endfunction()

option(BUILD_TESTING "Build and register relinker tests" OFF)
if(BUILD_TESTING)
    enable_testing()
    if(MINGW)
        get_filename_component(testCompilerDirectory "${CMAKE_CXX_COMPILER}" DIRECTORY)
        foreach(runtimeName IN ITEMS libgcc_s_seh-1.dll libstdc++-6.dll libwinpthread-1.dll)
            if(NOT EXISTS "${testCompilerDirectory}/${runtimeName}")
                message(FATAL_ERROR "Cannot locate MinGW runtime: ${runtimeName}")
            endif()
            file(COPY "${testCompilerDirectory}/${runtimeName}" DESTINATION "${ANYPS5_TEST_OUTPUT_DIRECTORY}")
        endforeach()
    endif()
endif()

option(ANYPS5_ENABLE_SPIRV_TOOLS "Enable SPIRV-Tools validation and optimization" OFF)
option(APS5_ENABLE_TIMING_LOG "Enable frame timing logging" OFF)
add_compile_definitions(APS5_ENABLE_TIMING_LOG=$<BOOL:${APS5_ENABLE_TIMING_LOG}>)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static -static-libgcc -static-libstdc++")

set(SDL_SHARED OFF CACHE BOOL "" FORCE)
set(SDL_STATIC ON CACHE BOOL "" FORCE)
set(SDL_TEST OFF CACHE BOOL "" FORCE)
set(SDL_VIDEO ON CACHE BOOL "" FORCE)
set(SDL_RENDER OFF CACHE BOOL "" FORCE)
set(SDL_X11 ON CACHE BOOL "" FORCE)
set(SDL_WAYLAND OFF CACHE BOOL "" FORCE)
set(SDL_OPENGL OFF CACHE BOOL "" FORCE)
set(SDL_OPENGLES OFF CACHE BOOL "" FORCE)
set(SDL_VULKAN ON CACHE BOOL "" FORCE)
set(SDL_JOYSTICK ON CACHE BOOL "" FORCE)
set(SDL_HAPTIC OFF CACHE BOOL "" FORCE)
set(SDL_HIDAPI ON CACHE BOOL "" FORCE)
set(SDL_SENSOR OFF CACHE BOOL "" FORCE)
set(SDL_STATIC_PIC ON CACHE BOOL "" FORCE)
set(ANYPS5_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "")
add_subdirectory(3rdparty/SDL2)
set(CMAKE_EXE_LINKER_FLAGS "${ANYPS5_EXE_LINKER_FLAGS}")
set(ARCHITECTURE x86_64)
add_subdirectory(3rdparty/ffmpeg-core)
if(MINGW)
    target_link_libraries(ffmpeg INTERFACE bcrypt ole32 user32 ws2_32)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(ffmpeg INTERFACE LINKER:--exclude-libs,ALL)
endif()

set(FT_DISABLE_ZLIB ON CACHE BOOL "" FORCE)
set(FT_DISABLE_BZIP2 ON CACHE BOOL "" FORCE)
set(FT_DISABLE_PNG ON CACHE BOOL "" FORCE)
set(FT_DISABLE_HARFBUZZ ON CACHE BOOL "" FORCE)
set(FT_DISABLE_BROTLI ON CACHE BOOL "" FORCE)
set(SKIP_INSTALL_ALL ON CACHE BOOL "" FORCE)
add_subdirectory(3rdparty/freetype EXCLUDE_FROM_ALL)
set_target_properties(freetype PROPERTIES POSITION_INDEPENDENT_CODE ON)

# mimalloc takes over the C runtime allocator of a relinked game's whole process on Windows: the
# executable's entry stub loads mimalloc.dll from the libraries' folder before any library, and its
# redirection module (mimalloc-redirect.dll, the prebuilt one mimalloc ships) patches ucrtbase's
# malloc family as it loads, before the modules' own initialization runs. Every module and the MinGW
# C++ runtime DLLs then allocate through it; new/delete are not overridden per module, so nothing
# depends on which allocator a DLL's own code was bound to. MIMALLOC_DISABLE_REDIRECT=1 keeps the C
# runtime allocator at run time; a libraries' folder without mimalloc.dll starts as before.
if(WIN32)
    set(MI_BUILD_SHARED ON CACHE BOOL "" FORCE)
    set(MI_BUILD_STATIC OFF CACHE BOOL "" FORCE)
    set(MI_BUILD_OBJECT OFF CACHE BOOL "" FORCE)
    set(MI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(MI_OVERRIDE ON CACHE BOOL "" FORCE)
    set(MI_WIN_REDIRECT ON CACHE BOOL "" FORCE)
    # A pointer mimalloc does not own (allocated before the redirection) is left alone instead of
    # being read as one of its blocks.
    set(MI_FREE_IS_CHECKED SHARED CACHE STRING "" FORCE)
    add_subdirectory(3rdparty/mimalloc EXCLUDE_FROM_ALL)
    if(MINGW)
        # A UCRT toolchain like MSYS2's UCRT64, which mimalloc recognizes by MSYSTEM only: initialize
        # through the C runtime's hooks as it does there.
        target_compile_definitions(mimalloc PRIVATE MI_MINGW_UCRT64=1)
        # Only system DLLs below it (libgcc and winpthreads linked in), so nothing of the MinGW runtime
        # initializes, and allocates, before it does. Its malloc/free fast paths read the thread from
        # the TEB; only its rarely used thread locals go through emulated TLS, whose own allocations
        # (and winpthreads') must not reach the redirected malloc: they take the process heap.
        target_link_options(mimalloc PRIVATE -static
                LINKER:--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free,--wrap=strdup,--wrap=_strdup)
        target_sources(mimalloc PRIVATE ${CMAKE_SOURCE_DIR}/core/libs/allocator/RuntimeHeap.c)
    endif()
endif()

add_subdirectory(core/shader/recompiler)
add_subdirectory(core/Decoder/Jpeg)
add_subdirectory(core/Decoder/Png)
add_subdirectory(core/libs)
add_subdirectory(core/relinker)

if(BUILD_TESTING)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME progress_report COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/tests/test_progress.py)
    endif()
endif()
