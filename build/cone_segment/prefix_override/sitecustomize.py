import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/woohyuck/Documents/Lecture_final/capstone_ws/install/cone_segment'
