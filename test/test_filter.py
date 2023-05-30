from EdiHeadyTrack import Filter

def test_init():
    filter = Filter()
    a = [round(elem, 2) for elem in list(filter.a)] 
    b = [round(elem, 2) for elem in list(filter.b)]
    assert list(a) == [1, -3.34, 4.24, -2.41, 0.52]
    assert list(b) == [0, 0, 0, 0, 0]  

def test_low_pass_butterworth():
    filter = Filter().low_pass_butterworth(fs=4000, 
                                           lowcut=160,
                                           order=4)
    a = [round(elem, 2) for elem in list(filter.a)] 
    b = [round(elem, 2) for elem in list(filter.b)]
    assert list(a) == [1, -3.34, 4.24, -2.41, 0.52]
    assert list(b) == [0, 0, 0, 0, 0]  

def test_apply():
    filter = Filter()
    import numpy as np
    signal = np.array([np.sin(x) for x in range(16)])
    filtered_signal = filter.apply(signal)
    filtered_signal = [round(elem, 0) for elem in filtered_signal]
    assert filtered_signal == [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 
                               -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                               0.0, 1.0, 1.0]