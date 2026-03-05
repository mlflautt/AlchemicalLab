/**
 * AlchemicalSynth - Terrain Synthesis Audio Processor
 * 
 * Based on Wave Terrain Synthesis approach
 * Combines CA, Fractals, and Neural systems for generative audio
 */

#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>

//==============================================================================
class AlchemicalSynthProcessor : public juce::AudioProcessor
{
public:
    //==============================================================================
    AlchemicalSynthProcessor();
    ~AlchemicalSynthProcessor() override;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;

    void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) override;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const juce::String getName() const override;
    bool acceptsMidi() const override;
    bool producesMidi() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;

    //==============================================================================
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    // Parameters
    float getParameterValue(int paramIndex) const { return *parameters[paramIndex]; }
    void setParameterValue(int paramIndex, float value);

private:
    //==============================================================================
    // Terrain synthesis state
    std::vector<float> terrainGrid;
    int terrainWidth = 128;
    int terrainHeight = 128;
    
    // Trajectory state
    float trajectoryX = 0.5f;
    float trajectoryY = 0.5f;
    float trajectoryPhase = 0.0f;
    
    // Envelope state
    enum { ENV_IDLE, ENV_ATTACK, ENV_DECAY, ENV_SUSTAIN, ENV_RELEASE };
    int envelopeStage = ENV_IDLE;
    float envelopeValue = 0.0f;
    
    // CA evolution
    int caGeneration = 0;
    int caEvolutionInterval = 1;
    
    // LFO
    float lfoPhase = 0.0f;
    
    // Parameters (automatable)
    std::vector<std::unique_ptr<juce::AudioParameterFloat>> parameters;
    
    // Parameter indices
    enum ParamIndex {
        PARAM_FREQUENCY = 0,
        PARAM_TERRAIN_TYPE,
        PARAM_CA_RULE,
        PARAM_FRACTAL_TYPE,
        PARAM_TRAJECTORY_SHAPE,
        PARAM_HYBRID_BLEND,
        PARAM_MEANDERANCE,
        PARAM_FILTER_CUTOFF,
        PARAM_FILTER_RES,
        PARAM_SATURATION,
        PARAM_ATTACK,
        PARAM_DECAY,
        PARAM_SUSTAIN,
        PARAM_RELEASE,
        PARAM_LFO_FREQ,
        PARAM_LFO_DEPTH,
        NUM_PARAMS
    };

    // Internal methods
    float sampleTerrain(float x, float y);
    float generateSample(float phase);
    void updateEnvelope();
    void evolveCA();
    void updateTrajectory();

    // Waveform generation
    enum WaveformType { SINE, SAW, SQUARE, TRIANGLE, SAMPLE_HOLD };
    WaveformType waveformType = SINE;

    // Trajectory types
    enum TrajectoryType { ELLIPSE, FIGURE8, LISSABOUS, MEANDER, SPIRAL, ROSE };
    TrajectoryType trajectoryType = ELLIPSE;

    // CA rule
    enum CARule { CONWAY, HIGHLIFE, BRIANS_BRAIN, DAY_NIGHT, SEEDS };
    CARule caRule = CONWAY;

    // Fractal type
    enum FractalType { PERLIN, WORLEY, MANDELBROT, JULIA };
    FractalType fractalType = PERLIN;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AlchemicalSynthProcessor)
};
