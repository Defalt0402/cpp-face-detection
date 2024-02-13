#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <vector>

using namespace cv;

CascadeClassifier faceClassifier;

Mat draw_bounding_box(Mat image, std::vector<Rect> rects){
    for (int i = 0; i < rects.size(); i++) {
        rectangle(image, Point(rects[i].x, rects[i].y), Point(rects[i].x + rects[i].width, rects[i].y + rects[i].height), Scalar(10, 25, 220), 2);
    }
    return image;
}

std::vector<Mat> get_faces(Mat frame){
    Mat greyFrame;
    cvtColor(frame, greyFrame, COLOR_BGR2GRAY);
    std::vector<Rect> faces;
    faceClassifier.detectMultiScale(greyFrame, faces, 1.1, 5);
    
    std::vector<Mat> faceImages;
    for (int i = 0; i < faces.size(); i++) {
        Rect coords = Rect(faces[i].x, faces[i].y, faces[i].x + faces[i].width, faces[i].y + faces[i].height);
        Mat face = frame(coords);
        faceImages.push_back(face);
    }
    
    return faceImages;
}

std::vector<Rect> get_face_rects(Mat frame){
    Mat greyFrame;
    cvtColor(frame, greyFrame, COLOR_BGR2GRAY);
    std::vector<Rect> faces;
    faceClassifier.detectMultiScale(greyFrame, faces, 1.1, 5);
    
    std::vector<Mat> faceImages;
    for (int i = 0; i < faces.size(); i++) {
        faces.push_back(Rect(faces[i].x, faces[i].y, faces[i].x + faces[i].width, faces[i].y + faces[i].height));
    }

    return faces;
}

Mat process_for_model(Mat image) {
    Mat greyImage;
    Mat processedImage;
    cvtColor(image, greyImage, COLOR_BGR2GRAY);
    resize(greyImage, processedImage, cv::Size(48,48));
    processedImage.convertTo(processedImage, CV_32FC3, 1.f/255);
    return processedImage;
}

int main(int argc, char *argv[]){
    Mat img;
    Mat detectedFaces;
    std::vector<Mat> faceImages;
    std::vector<Mat> processedFaces;
    std::vector<Rect> faceRects;
    std::vector<std::string> emotionPredictions;

    dnn::Net model = dnn::readNetFromTensorflow("");
    std::map<int, std::string> classIdStringMap = {{0, "Angry"}, 
                                                    {1, "Disgust"}, 
                                                    {2, "Fear"}, 
                                                    {3, "Happy"}, 
                                                    {4, "Sad"}, 
                                                    {5, "Surprise"}, 
                                                    {6, "Neutral"}};

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

    while (true) {
        video.read(img);

        if (img.empty()) {
            std::cerr << "Error, unable to capture frame" << std::endl;
            continue;
        }
        
        faceImages = get_faces(img);
        if (faceImages.size() > 0) {
            faceRects = get_face_rects(img);

            for (int i = 0; i < faceImages.size(); i++) {
                processedFaces.push_back(process_for_model(faceImages[i]));
            }

            for (int i = 0; i < processedFaces.size(); i++) {
                Mat blob = dnn::blobFromImage(processedFaces[i]);
                model.setInput(blob);

                Mat probability = model.forward();

                Mat sortedProbabilities;
                Mat sortedIds;
                sort(probability.reshape(1, 1), sortedProbabilities, cv::SORT_DESCENDING);
                sortIdx(probability.reshape(1, 1), sortedIds, cv::SORT_DESCENDING);

                float topProbability = sortedProbabilities.at<float>(0);
                int topClassId = sortedIds.at<int>(0);

                std::string className = classIdStringMap.at(topClassId);
                std::string result = className + ": " + std::to_string(topProbability * 100) + "%";

                // Put on end of result vector
                emotionPredictions.push_back(result);
            }

            for (int i = 0; i < emotionPredictions.size(); i++) {
                Rect face = faceRects[i];
                putText(img, emotionPredictions[i], Point(face.x, face.y - 15), FONT_HERSHEY_PLAIN, 1, Scalar(15, 20, 220), 2);
            }
        }
        
        
        img = draw_bounding_box(img, faceRects);
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