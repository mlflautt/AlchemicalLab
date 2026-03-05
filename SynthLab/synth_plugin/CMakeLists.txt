cmake_minimum_required(VERSION 3.15)
project(AlchemicalSynth VERSION 1.0.0)

# JUCE settings
set(JUCE_VERSION 7.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find JUCE (assumes JUCE is cloned as submodule or found in system)
find_package(JUCE CONFIG REQUIRED)

# Add source files
set(SOURCE_FILES
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
    Source/TerrainEngine.cpp
    Source/CAEngine.cpp
    Source/FractalGenerator.cpp
    Source/Trajectory.cpp
    Source/PresetManager.cpp
)

# Create the plugin
juce_add_plugin(AlchemicalSynth
    PRODUCT_NAME "AlchemicalSynth"
    COMPANY_NAME "AlchemicalLab"
    COMPANY_WEBSITE "https://github.com/mlflautt/AlchemicalLab"
    COMPANY_EMAIL "mitchell@alchemical.ai"
    
    # Plugin characteristics
    FORMATS AU VST3
    PLUGIN_MANUFACTURER_CODE Alch
    PLUGIN_CODE AlchS
    
    # Audio configuration
    NEEDS_MIDI_INPUT NO
    NEEDS_MIDI_OUTPUT NO
    IS_SYNTH YES
    
    # VST3 specific
    VST3_CATEGORIES Synth
)

# Link required modules
target_link_libraries(AlchemicalSynth
    PRIVATE
    juce::juce_core
    juce::juce_audio_devices
    juce::juce_audio_processors
    juce::juce_audio_utils
    juce::juce_dsp
)

# Copy presets
if(EXISTS ${CMAKE_SOURCE_DIR}/Resources/Presets)
    file(COPY ${CMAKE_SOURCE_DIR}/Resources/Presets DESTINATION ${CMAKE_BINARY_DIR}/Presets)
endif()

# Enable debugging symbols for development
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(AlchemicalSynth PRIVATE DEBUG=1)
endif()
