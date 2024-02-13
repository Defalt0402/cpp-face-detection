#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>

using namespace cv;

int main(int argc, char *argv[]){
    // Declare matrix to store frame in
    Mat img;

    // Stores output from waitkey function
    // keys are ascii values
    char key;

    // Declare video object
    // Gets input from camera 0
    VideoCapture video = VideoCapture(0);

    namedWindow("Detection", WINDOW_NORMAL);

    // Gets the next frame of the video
    video.read(img);

    imshow("Detection", img);

    while (true) {
        video.read(img);
        imshow("Detection", img);
        key = waitKey(1);

        // Check if ESC (33) or q (113) is pressed to allow quit
        if (key == 33 || key == 113) {
            break;
        }
    }

    video.release();
    destroyAllWindows();
}