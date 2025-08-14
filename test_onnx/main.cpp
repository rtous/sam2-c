//NOTE: This test uses a minimal image library (stb) instead of OpenCV 

//Example explained here: https://onnxruntime.ai/docs/tutorials/mnist_cpp.html
//Check model info here: https://github.com/onnx/models/tree/main/validated/vision/classification/mnist
//MNIST’s input is a {1,1,28,28} shaped float tensor, which is basically a 28x28 floating point grayscale image (0.0 = background, 1.0 = foreground).
//MNIST’s output is a simple {1,10} float tensor that holds the likelihood weights per number. The number with the highest value is the model’s best guess.
//The MNIST structure uses std::max_element to do this and stores it in result_:

#include <stdio.h>  
#include "mnist.h"
#include "util.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "stb_image_write.h"
 
void print_MNIST_digit(unsigned char image_stb[], int nx, int ny) 
{
  //row,col -> image y,x (the intuitive way)
  for (unsigned i = 0; i < nx; i++) {
    for (unsigned j = 0; j < ny; j++) {
        //printf("%u\n", image_stb[i*ny+j]);
        if (image_stb[(i*ny+j)*3]>30) 
          printf("# "); 
        else
          printf("  ");
    }
    printf("\n");
  }
  printf("\n");
}

/*void print_MNIST_digit(cv::Mat img) 
{
  //opencv Mat row,col -> image y,x (the intuitive way)
  for (unsigned i = 0; i < 28; i++) {
    for (unsigned j = 0; j < 28; j++) {
      if (img.at<uchar>(i, j)>30)
        printf("# ");
      else
        printf("  ");
    }
    printf("\n");
  }
  printf("\n");
}*/

int main()
{
  printf("Hello world\n");

  //1) Prepare the ONNX model (file mnist.onnx harcoded in mnist.h)
  std::unique_ptr<MNIST> mnist_;
  mnist_ = std::make_unique<MNIST>();

  //2) Read input image
  int nx, ny, nc;
  auto image_stb = stbi_load("mnist.jpg", &nx, &ny, &nc, 3);
  //WARNING: Even if nc==1 it stores 3 bytes per pixel
  if (!image_stb) {
        fprintf(stderr, "%s: failed to load '%s'\n", __func__, "mnist.jpg");
        return 1;
  }
  printf("Image successfully read. Dimensions: %d rows, %d columns, %d channels.\n", nx, ny, nc);
  print_MNIST_digit(image_stb, nx, ny);

  //3) Convert the stb image to ONNX tensor
  //In typdef MSNIT in mnist.h: 
  //  std::array<float, width_ * height_> input_image_{};
  float* output = mnist_->input_image_.data();
  std::fill(mnist_->input_image_.begin(), mnist_->input_image_.end(), 0.f);
  for (unsigned y = 0; y < MNIST::height_; y++) {
    for (unsigned x = 0; x < MNIST::width_; x++) {
      output[y * MNIST::height_ + x] = image_stb[(y*ny+x)*3]/(float)255;
    }
  }

  //Run the model
  mnist_->Run();
  
  //Results
  auto least = *std::min_element(mnist_->results_.begin(), mnist_->results_.end());
  auto greatest = mnist_->results_[mnist_->result_];
  auto range = greatest - least;

  for (unsigned i = 0; i < 10; i++) {
    int y = 16 * i;
    float result = mnist_->results_[i];
    printf("%2d: %d.%02d\n", i, int(result), abs(int(result * 100) % 100));
  }
  return 0;
}
