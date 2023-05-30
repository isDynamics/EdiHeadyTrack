from EdiHeadyTrack.video import Video

TEST_FILE = 'test/resources/testvidshort.mp4'
TEST_VIDEO = Video(TEST_FILE)

def test_open_vid():
    cap = TEST_VIDEO._open_vid()
    import cv2
    assert type(cap) == cv2.VideoCapture

def test_create_writer():
    TEST_VIDEO.create_writer()
    import cv2
    assert type(TEST_VIDEO.writer) == cv2.VideoWriter

def test_get_dim():
    width, height = TEST_VIDEO.get_dim()
    assert width == 1280
    assert height == 720

def test_get_length():
    total_frames = TEST_VIDEO.get_length()
    assert total_frames == 76

def test_get_fps():
    fps = TEST_VIDEO.get_fps()
    assert fps == 240