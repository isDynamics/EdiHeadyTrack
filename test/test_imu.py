from EdiHeadyTrack.imu import Wax9

def test_Wax9_extract_from_file():
    wax9 = Wax9('resources/example_imu.csv', time_offset=-59.335, id='WAX-9')
    assert round(wax9.velocity['yaw'][0], 2) == -5.67
    assert round(wax9.acceleration['pitch'][0], 2) == 0.2