import warnings

def setup_warnings():
    warnings.simplefilter(action='ignore', category=Warning)
