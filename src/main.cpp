#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace cv::ml;
using namespace cv::dnn;

// Creates a SVM Model.
Ptr<SVM> createSVMModel()
{
    Ptr<SVM> model = SVM::create();
    model->setKernel(model->POLY);
    model->setType(model->C_SVC);
    model->setDegree(2);
    return model;
}

// Load the training data. Images are stored in features Mat and the classification of the image in labels Mat.
// The images are loaded in grayscale and the size of each image is 300 x 300.
void loadTrainData(std::string imagesPath, std::string label, Mat &features, Mat &labels)
{
    Mat imageGray;
    Mat imageResized;
    Mat image32F;
    Mat imageBlur;
    Mat flat;
    Mat normalized;
    for (int i = 0; i < 600; i++)
    {
        std::string imageName = imagesPath + "\\" + label + "s\\" + label + "." +
                                std::to_string(i + 1) + ".jpg";
        imageGray = imread(imageName, cv::IMREAD_GRAYSCALE);
        resize(imageGray, imageResized, Size(310, 310));
        medianBlur(imageResized, imageBlur, 3);
        normalize(imageBlur, normalized, 0, 255, NORM_MINMAX, -1);
        normalized.convertTo(image32F, CV_32F);
        flat = image32F.reshape(1, 1);
        features.push_back(flat);
        labels.push_back(label == "cat" ? 1 : 0);
    }
}

// Training de SVM model using the features and labels loaded in loadTrainData function.
float trainModel(Ptr<SVM> model, Mat &features, Mat &labels)
{
    Ptr<TrainData> trainData = TrainData::create(features, ROW_SAMPLE, labels);
    trainData->setTrainTestSplitRatio(0.2, true);
    model->train(trainData);
    Mat predict;
    float error = model->calcError(trainData, true, predict);
    return error;
}

// Function to test our SVM model. Receives an image and return the classification. 1 for cat and 0 for human.
float predictModel(Ptr<SVM> model, Mat &image)
{
    Mat imageResized;
    Mat flat;
    Mat image32F;
    resize(image, imageResized, Size(300, 300));
    imageResized.convertTo(image32F, CV_32F);
    flat = image32F.reshape(1, 1);
    model->save("..\\model.xml");
    float result = model->predict(flat);
    return result;
}

int main(int argc, char const *argv[])
{
    Ptr<SVM> model = createSVMModel();
    std::string imagesPath = "..\\images\\classification\\train";
    Mat features;
    Mat labels;
    std::cout << "Loading cat images..." << std::endl;
    loadTrainData(imagesPath, "cat", features, labels);
    std::cout << "Loading human images..." << std::endl;
    loadTrainData(imagesPath, "human", features, labels);
    std::cout << "Number of features loaded: " << features.size() << std::endl;
    std::cout << "Number of labels loaded: " << labels.size() << std::endl;
    std::cout << "Training SVM model..." << std::endl;
    float error = trainModel(model, features, labels);
    std::cout << "Error: " << error << std::endl;
    return EXIT_SUCCESS;
}