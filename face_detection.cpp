#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <vector>

using namespace cv;

CascadeClassifier faceClassifier;

Mat detect_bounding_box(Mat frame){
    Mat greyFrame;
    cvtColor(frame, greyFrame, COLOR_BGR2GRAY);
    std::vector<Rect> faces;
    faceClassifier.detectMultiScale(greyFrame, faces, 1.1, 5);
    for (int i = 0; i < faces.size(); i++) {
        rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(10, 25, 220), 2);
    }
    return frame;
}

int main(int argc, char *argv[]){
    Mat img;
    Mat detectedFaces;

    // Stores output from waitkey function
    // keys are ascii values
    char key;

    // Declare video object
    // Gets input from camera 0
    VideoCapture video = VideoCapture(0);

    // Check if the camera opened successfully
    if (!video.isOpened()) {
        std::cerr << "Error: Unable to open the camera" << std::endl;
        return -1;
    }

    if (!faceClassifier.load("/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")) {
        std::cerr << "Error loading the cascade classifier" << std::endl;
        return -1;
    }


    namedWindow("Detection", WINDOW_NORMAL);

    // Gets the next frame of the video
    video.read(img);

    imshow("Detection", img);

    while (true) {
        video.read(img);

        if (img.empty()) {
            std::cerr << "Error, unable to capture frame" << std::endl;
            continue;
        }
        
        detectedFaces = detect_bounding_box(img);
        imshow("Detection", detectedFaces);
        key = waitKey(1);

        // Check if ESC (33) or q (113) is pressed to allow quit
        if (key == 33 || key == 113) {
            break;
        }
    }

    video.release();
    destroyAllWindows();
}