from src.data.vst.param_spec import (
    CategoricalParameter,
    ContinuousParameter,
    DiscreteLiteralParameter,
    NoteDurationParameter,
    ParamSpec,
)

SURGE_VIM_PARAM_SPEC = ParamSpec(
    [
    #Amplitude ADSR envelope
        ContinuousParameter(name="a_amp_eg_attack", min=0.0, max=0.77),  
        ContinuousParameter(name="a_amp_eg_decay", min=0.0, max=0.77),  # min increased to ensure audibility
        ContinuousParameter(name="a_amp_eg_sustain", min=0.0, max=1.0),
        ContinuousParameter(
            name="a_amp_eg_release", min=0.0, max=0.77
        ),  # max around 4s    
        # CategoricalParameter(
        #     name="a_amp_eg_envelope_mode",
        #     values=["Digital"],
        #     raw_values=[0.25],
        #     encoding="onehot",
        # ),

    # FILTER 
        ContinuousParameter(
            name="a_filter_balance",
            min=0.0,
            max=1.0,
            constant_val_p=0.0,
            constant_val=0.0,
        ),
        # CategoricalParameter(
        #     name="a_filter_configuration",
        #     values=[
        #         "Serial 1",
        #         "Dual 1",
        #     ],
        #     raw_values=[
        #         0.038,
        #         0.4295,
        #     ],
        #     encoding="onehot",
        # ),
        ContinuousParameter(
            name="a_filter_eg_attack", min=0.0, max=0.1
        ),  
        ContinuousParameter(
            name="a_filter_eg_decay", min=0.2, max=0.77 # min increased
        ),  # max around 4s
        ContinuousParameter(
            name="a_filter_eg_release", min=0.0, max=0.77
        ),  # max around 4s
        ContinuousParameter(name="a_filter_eg_sustain", min=0.0, max=1.0),
        # CategoricalParameter(
        #     name="a_filter_eg_envelope_mode",
        #     values=["Digital"],
        #     raw_values=[0.25],
        #     encoding="onehot",
        # ),
        ContinuousParameter(name="a_highpass", min=0.0, max=0.3), # max reduced to avoid total silence


        # -- Filter 1
        ContinuousParameter(name="a_filter_1_cutoff", min=0.2, max=1.0), # min increased (approx 100hz)
        ContinuousParameter(name="a_filter_1_feg_mod_amount", min=0.0, max=1.0),
        ContinuousParameter(name="a_filter_1_resonance", min=0.0, max=1.0),
        CategoricalParameter(
            name="a_filter_1_type",
            values=[
                "LP 12 dB",
                "LP 24 dB",
            ],
            raw_values=[
                0.035500000000000004,
                0.0655,
            ],
            encoding="onehot",
        ),


        # -- Filter 2
        ContinuousParameter(name="a_filter_2_cutoff", min=0.0, max=1.0),
        ContinuousParameter(name="a_filter_2_feg_mod_amount", min=0.0, max=1.0),
        ContinuousParameter(name="a_filter_2_resonance", min=0.0, max=1.0),
        CategoricalParameter(
            name="a_filter_2_type",
            values=[
                "HP 12 dB",
                "HP 24 dB",
                "BP 12 dB",
                "N 12 dB",
                # "FX Comb +",
                "BP 24 dB",
                "N 24 dB",
                # "FX Comb -",
                # "FX Allpass",
            ],
            raw_values=[
                0.1255,
                0.15500000000000003,
                0.185,
                0.21500000000000002,
                # 0.2455,
                0.6955,
                0.7255,
                # 0.7555000000000001,
                # 0.7855000000000001,
            ],
            encoding="onehot",
        ),

    # OSCILLATORS            
        # Oscillator 1 parameters

        # CategoricalParameter(
        #     name="a_osc_1_type",
        #     values=["Modern"],
        #     raw_values=[0.7083], # Derived from probing (Indices 0 and 8)
        #     encoding="onehot"
        # ), 
        CategoricalParameter(
            name="a_osc_1_mute",
            values=[False, True],
            raw_values=[0.2505, 0.7505],
            weights=[0.9, 0.1],
            encoding="onehot",
        ),
        CategoricalParameter(
            name="a_osc_1_octave",
            values=[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            raw_values=[
                0.044,
                0.17049999999999998,
                0.3355,
                0.5005,
                0.6655,
                0.8305,
                0.9565,
            ],
            weights=[
                1.0,
                1.0,
                1.0,
                6.0,
                1.0,
                1.0,
                1.0,
            ],
            encoding="onehot",
        ),
        ContinuousParameter(
            name="a_osc_1_pitch", min=0.49, max=0.51, constant_val_p=0.5, constant_val=0.5
        ),
        
        CategoricalParameter(
            name="a_osc_1_route",
            values=["Filter 1"],
            raw_values=[0.1265],
            encoding="onehot",
        ),
        
        # ContinuousParameter(name="a_osc_1_shape", min=0.0, max=1.0),
        # ContinuousParameter(name="a_osc_1_width_1", min=0.0, max=1.0),
        # ContinuousParameter(name="a_osc_1_width_2", min=0.0, max=1.0),
        ContinuousParameter(name="a_osc_1_sawtooth", min=0.0, max=1.0),
        ContinuousParameter(name="a_osc_1_width", min=0.0, max=1.0),
        ContinuousParameter(name="a_osc_1_sync", min=0.0, max=1.0),
        ContinuousParameter(
            name="a_osc_1_unison_detune", 
            min=0.0, 
            max=0.3, 
        ),
        # CategoricalParameter(
        #     name="a_osc_1_unison_voices",
        #     values=["1 voices"],
        #     raw_values=[0.0195],
        #     encoding="onehot",
        # ),
        ContinuousParameter(name="a_osc_1_volume", min=0.6, max=1.0), # min increased (approx -10dB)
        ContinuousParameter(name="a_osc_1_pulse", min=0.0, max=1.0),
        ContinuousParameter(name="a_osc_1_triangle", min=0.0, max=1.0),
    
        # Oscillator 2 parameters

        # -- Oscillator 2
        CategoricalParameter(
            name="a_osc_2_mute",
            values=[False, True],
            raw_values=[0.2505, 0.7505],
            weights=[0.5, 0.5],
            encoding="onehot",
        ),
        CategoricalParameter(
            name="a_osc_2_octave",
            values=[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            weights=[
                1.0,
                1.0,
                1.0,
                6.0,
                1.0,
                1.0,
                1.0,
            ],
            encoding="onehot",
        ),
        ContinuousParameter(
            name="a_osc_2_pitch", min=0.0, max=1.0, constant_val_p=0.5, constant_val=0.5
        ),
        CategoricalParameter(
            name="a_osc_2_route",
            values=["Both", "Filter 2"],
            raw_values=[0.5005, 0.874],
            encoding="onehot",
        ),
        ContinuousParameter(name="a_osc_2_sawtooth", min=0.0, max=1.0),
        ContinuousParameter(name="a_osc_2_width", min=0.0, max=1.0),
        ContinuousParameter(name="a_osc_2_sync", min=0.0, max=1.0),
        ContinuousParameter(
            name="a_osc_2_unison_detune",
            min=0.0,
            max=1.0,
        ),
        # CategoricalParameter(
        #     name="a_osc_2_unison_voices",
        #     values=[
        #         "1 voice",
        #         "2 voices",
        #         "3 voices",
        #         "4 voices",
        #     ],
        #     raw_values=[
        #         0.0195,
        #         0.07150000000000001,
        #         0.137,
        #         0.203,
        #     ],
        #     weights=[3.0, 1, 1, 1],
        #     encoding="onehot",
        # ),
        ContinuousParameter(name="a_osc_2_volume", min=0.0, max=1.0),
        ContinuousParameter(name="a_osc_2_pulse", min=0.0, max=1.0),
        ContinuousParameter(name="a_osc_2_triangle", min=0.0, max=1.0),
        
        # Oscillator 3 OFF
        CategoricalParameter(
            name="a_osc_3_mute",
            values=[False, True],
            raw_values=[0.2505, 0.7505],
            weights=[0, 1],
            encoding="onehot",)
            ,

        # Noise
        ContinuousParameter(name="a_noise_color", min=0.0, max=1.0),
        CategoricalParameter(
            name="a_noise_mute",
            values=[False, True],
            raw_values=[0.2505, 0.7505],
            weights=[0.33, 0.67],
            encoding="onehot",
        ),
        CategoricalParameter(
            name="a_noise_route",
            values=["Filter 1", "Both", "Filter 2"],
            raw_values=[0.1265, 0.5005, 0.874],
            encoding="onehot",
            weights=[0.8, 0.1, 0.1],
        ),
        ContinuousParameter(
            name="a_noise_volume", min=0.0, max=1.0, constant_val_p=0.67
        ),
    ],
    [
        DiscreteLiteralParameter(
            name="pitch",
            min=36,
            max=72,
        ),
        NoteDurationParameter(name="note_start_and_end", max_note_duration_seconds=4.0),
    ]
)
