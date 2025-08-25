Project Description
Facial detection and recognition system built on the Jetson Nano. 
The application streams IP camera feeds via GStreamer, performs face detection using OpenCV Haar cascades, and enables face recognition. 
A Flask web app provides functionality to save detected faces, assign or edit categories, and log all entries with timestamps in a database.

Camera Connectivity
The RTSP pipeline authenticates with the IP camera using a username, password, and port number (default: 554) for connectivity.

Environment
Hardware: NVIDIA Jetson Nano, DAHUA IP Vision 8mp Camera
OS: L4T (Linux for Tegra)
CUDA: Enabled for hardware acceleration
