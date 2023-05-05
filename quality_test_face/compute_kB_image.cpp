#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

int main() {
    Mat img = imread("path/to/image.jpg");
    int file_size = fs::file_size("path/to/image.jpg") / 1024; // chuyển đổi kích thước sang kB
    cout << "Dung lượng của ảnh là " << file_size << " kB" << endl;
    return 0;
}