# africa_tts_frontend: text -> Africa TTS IPA (africa-g2p C library + espeak-ng for English).
#
#   cmake -S mobile/frontend -B build/frontend && cmake --build build/frontend -j
#
# africa-g2p's native library is fetched at the commit the models were trained with.
# espeak-ng: found on the system by default (libespeak-ng-dev); for Android/iOS build espeak-ng
# for the target and pass -DESPEAK_NG_INCLUDE_DIR=... -DESPEAK_NG_LIBRARY=..., or
# -DATT_WITH_ESPEAK=OFF to drop English (and the GPL dependency).
cmake_minimum_required(VERSION 3.16)
project(africa_tts_frontend C)

set(CMAKE_C_STANDARD 11)
option(ATT_WITH_ESPEAK "English via espeak-ng (GPL-3.0-or-later)" ON)
set(AFRICA_G2P_COMMIT 76b53a8fa2a688812e369837d179d4710c111a7a CACHE STRING "africa-g2p commit")

include(FetchContent)
FetchContent_Declare(africa_g2p
  GIT_REPOSITORY https://github.com/AfriSpeech/africa-g2p.git
  GIT_TAG ${AFRICA_G2P_COMMIT}
  SOURCE_SUBDIR native)
FetchContent_MakeAvailable(africa_g2p)

add_library(africa_tts_frontend africa_tts_frontend.c)
target_include_directories(africa_tts_frontend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(TARGET afrig2p_static)
  target_link_libraries(africa_tts_frontend PUBLIC afrig2p_static)
else()
  target_link_libraries(africa_tts_frontend PUBLIC afrig2p)
endif()

if(ATT_WITH_ESPEAK)
  find_path(ESPEAK_NG_INCLUDE_DIR espeak-ng/speak_lib.h)
  find_library(ESPEAK_NG_LIBRARY espeak-ng)
  if(NOT ESPEAK_NG_INCLUDE_DIR OR NOT ESPEAK_NG_LIBRARY)
    message(FATAL_ERROR "espeak-ng not found: install libespeak-ng-dev, set ESPEAK_NG_* or -DATT_WITH_ESPEAK=OFF")
  endif()
  target_include_directories(africa_tts_frontend PRIVATE ${ESPEAK_NG_INCLUDE_DIR})
  target_link_libraries(africa_tts_frontend PUBLIC ${ESPEAK_NG_LIBRARY})
else()
  target_compile_definitions(africa_tts_frontend PRIVATE ATT_NO_ESPEAK)
endif()

add_executable(att_ipa att_ipa.c)
target_link_libraries(att_ipa PRIVATE africa_tts_frontend)

# Android: libafrica_tts_frontend_jni.so for org.afrispeech.africatts.AfricaTtsFrontend.
option(ATT_BUILD_JNI "Build the JNI binding (Android)" ${ANDROID})
if(ATT_BUILD_JNI)
  if(NOT ANDROID)
    find_package(JNI REQUIRED)
  endif()
  add_library(africa_tts_frontend_jni SHARED jni/att_jni.c)
  target_include_directories(africa_tts_frontend_jni PRIVATE ${JNI_INCLUDE_DIRS})
  target_link_libraries(africa_tts_frontend_jni PRIVATE africa_tts_frontend)
endif()
