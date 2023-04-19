import EdiHeadyTrack.logging
import matplotlib.pyplot as plt 


def test_log_show_false():
    assert EdiHeadyTrack.logging.log(show=False) == None

def test_log_show_true():
    assert EdiHeadyTrack.logging.log(show=True) == None