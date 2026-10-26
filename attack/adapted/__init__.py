"""Six selected Person-in-WiFi-3D CSI attack adaptations.

The main train_backdoor.py entry point remains A1 only. These operators are
provided for method inspection and use by an experiment-specific controller.
"""

METHODS = ('wanet_source', 'ftrojan', 'fiba', 'bpp_source',
           'smooth_lf_csi', 'tsba_adapted')

def build_trigger(name, training_data=None):
    if name in METHODS[:4]:
        if training_data is None:
            raise ValueError('TRAIN dataset required for source adapters')
        from attack.adapted.piw_native import GroupedTrigger
        return GroupedTrigger(name, training_data)
    if name == 'smooth_lf_csi':
        from attack.adapted.piw_extra import PiWSmoothLFTrigger
        return PiWSmoothLFTrigger()
    if name == 'tsba_adapted':
        from attack.adapted.piw_extra import PiWTSBATrigger
        return PiWTSBATrigger()
    raise ValueError(f'unknown adapted method: {name}')
