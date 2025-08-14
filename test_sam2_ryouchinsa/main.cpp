#include <stdio.h>  
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp> 
#include <opencv2/core/utils/filesystem.hpp>
#include <print>
#include <iostream> 

//NOTE: The data() function returns a pointer to the block of memory where a vector's elements are stored.

int tensorByName(std::vector<Ort::Value> &tensors, std::vector<const char*> names, char* name_queried) {
    int i;
    for (i=0; i<names.size() && strcmp(names[i], name_queried) != 0; i++) {
        printf("Checked %s != %s\n", names[i], name_queried);
    }
    if (i<names.size()) {
        return i;
    } else {
        printf("ERROR, tensor name %s not found. \n", name_queried);
        exit(-1);
    }
}

struct Node{
    char* name = nullptr;
    std::vector<int64_t> dim; // batch,channel,height,width
};

void preprocess(cv::Mat &image, std::vector<cv::Mat> &input_images){
    cv::Mat image_ = image.clone();
    // cv::subtract(image, cv::Scalar(0.406, 0.456, 0.485), image_);
    // cv::divide(image_, cv::Scalar(0.225, 0.224, 0.229), image_);
    std::vector<cv::Mat> mats{image_};
    //OpenCV function to prepare image for an NN. Mat into a 4-dimensional array/blob
    cv::Mat blob = cv::dnn::blobFromImages(mats, 1/255.0,cv::Size(1024,1024), cv::Scalar(0, 0, 0), true, false);
    input_images.clear();
    //Add preprocessed frame to input_images
    input_images.emplace_back(blob);
}

void postprocess(std::vector<Ort::Value> &output_tensors, cv::Mat *ori_img, cv::Rect prompt_box, cv::Point prompt_point){
    float* output =  output_tensors[0].GetTensorMutableData<float>();
    cv::Mat outimg(ori_img->size(),CV_32FC1, output);
    cv::Mat dst;
    outimg.convertTo(dst, CV_8UC1, 255);
    cv::threshold(dst,dst,0,255,cv::THRESH_BINARY);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(dst, dst, cv::MORPH_OPEN, element);
    std::vector<std::vector<cv::Point>> contours; // 不一定是1
    cv::findContours(dst, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    int idx = -1;
    cv::Rect min_dis_rect;
    double min_dis = std::numeric_limits<double>::max();
    // 计算与 A 中心距离最近的 bbox
    for (size_t i = 0;i<contours.size();i++) {
        cv::Rect bbox = cv::boundingRect(contours[i]);
        cv::Point center(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
        cv::Point A_center(prompt_box.x + prompt_box.width / 2, prompt_box.y + prompt_box.height / 2);
        double distance = cv::norm(center - A_center);
        if (distance < min_dis) {
            min_dis = distance;
            min_dis_rect = bbox;
            idx = i;
        }
    }
    if (!min_dis_rect.empty()) {
        prompt_box = min_dis_rect;
        prompt_point.x = min_dis_rect.x + min_dis_rect.width/2;
        prompt_point.y = min_dis_rect.y + min_dis_rect.height/2;
        cv::drawContours(*ori_img, contours, idx, cv::Scalar(50,250,20),1,cv::LINE_AA);
        cv::rectangle(*ori_img, prompt_box,cv::Scalar(0,0,255),2);
    }
}

void postprocess2(std::vector<Ort::Value> &output_tensors, cv::Mat *ori_img, cv::Rect prompt_box, cv::Point prompt_point){
    float* output =  output_tensors[0].GetTensorMutableData<float>();
    cv::Mat outimg(ori_img->size(),CV_32FC1, output);
    cv::imshow("Image", outimg);
    cv::waitKey(0);
}

void load_onnx_info(Ort::Session* session,std::vector<Node>& input,std::vector<Node>& output,std::string onnx="default.onnx"){
    Ort::AllocatorWithDefaultOptions allocator;
    // 模型输入信息
    for (size_t index = 0; index < session->GetInputCount(); index++) {
        Ort::AllocatedStringPtr input_name_Ptr = session->GetInputNameAllocated(index, allocator);
        Ort::TypeInfo input_type_info = session->GetInputTypeInfo(index);
        auto input_dims = input_type_info.GetTensorTypeAndShapeInfo().GetShape();
        Node node;
        node.dim = input_type_info.GetTensorTypeAndShapeInfo().GetShape();
        const char* name = input_name_Ptr.get();
        size_t name_length = strlen(name) + 1;
        node.name = new char[name_length];
        strcpy(node.name, name);
        input.push_back(node);
    }
    // 模型输出信息
    for (size_t index = 0; index < session->GetOutputCount(); index++) {
        Ort::AllocatedStringPtr output_name_Ptr = session->GetOutputNameAllocated(index, allocator);
        Ort::TypeInfo output_type_info = session->GetOutputTypeInfo(index);
        Node node;
        node.dim = output_type_info.GetTensorTypeAndShapeInfo().GetShape();
        const char* name = output_name_Ptr.get();
        size_t name_length = strlen(name) + 1;
        node.name = new char[name_length];
        strcpy(node.name, name);
        output.push_back(node);
    }
    // 打印日志
    printf("***************%s***************\n", onnx.c_str());
    for(auto &node:input){
        std::string dim_str = "[";
        for (size_t i = 0; i < node.dim.size(); ++i) {
            dim_str += std::to_string(node.dim[i]);
            if (i != node.dim.size() - 1) dim_str += ",";
        }
        dim_str += "]";
        printf("input_name= [%s] ===> %s\n", node.name, dim_str.c_str());
    }
    for(auto &node:output){
        std::string dim_str = "[";
        for (size_t i = 0; i < node.dim.size(); ++i) {
            dim_str += std::to_string(node.dim[i]);
            if (i != node.dim.size() - 1) dim_str += ",";
        }
        dim_str += "]";
        printf("output_name= [%s] ==> %s\n", node.name, dim_str.c_str());
    }
    printf("************************************\n");
}

int main()
{
  int frame_number = 0;

  // 1) Create CPU
  printf("Allocating memory...\n");
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  // 2) Read input image (or input video)
  printf("Reading input image...\n");
  std::vector<cv::Mat> input_images;
  cv::Mat image = cv::imread("band.jpg", cv::IMREAD_COLOR);
  if(image.empty())
  {
    printf("Cannot read the input image. \n");
    return 1;
  }
  int ori_img_cols = image.size[1];
  int ori_img_rows = image.size[0];
  printf("Successfully read (cols=%d, rows=%d)...\n", ori_img_cols, ori_img_rows);

  //3) preprocess input image (opencv to onnx format)
  printf("Preprocessing input image...\n");
  preprocess(image, input_images); //Add image to the vector (as is the first will be in pos [0])
  printf("Successfully preprocessed input image.\n");

  /////////////////
  // IMAGE ENCODER
  /////////////////
  //input_name= [input] ===> [1,3,1024,1024]
  //output_name= [image_embeddings] ==> [1,256,64,64]
  //output_name= [high_res_features1] ==> [1,32,256,256]
  //output_name= [high_res_features2] ==> [1,64,128,128]

  //1.1) Create ORT Session (image incoder)
  printf("Creating ORT Session (image incoder)...\n");
  Ort::Env img_encoder_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING,"img_encoder");
  Ort::SessionOptions img_encoder_options = Ort::SessionOptions();
  img_encoder_options.SetIntraOpNumThreads(2);
  Ort::Session* img_encoder_session = new Ort::Session(img_encoder_env, "checkpoints/ryouchinsa/sam2.1_tiny_preprocess.onnx", img_encoder_options);
  //Ort::Session* img_encoder_session = new Ort::Session(img_encoder_env, "checkpoints/ailia/large/image_encoder_hiera_l.onnx", img_encoder_options);
  printf("ORT session created (image incoder).\n");


  //1.2) Obtain information about the ONNX model (image incoder)
  printf("Obtaining information about the ONNX model (image incoder)...\n");
  std::vector<Node> img_encoder_input_nodes;
  std::vector<Node> img_encoder_output_nodes;
  load_onnx_info(img_encoder_session, img_encoder_input_nodes, img_encoder_output_nodes,"img_encoder");
  printf("Information obtained (image incoder).\n");
  

  //1.3) PREPARE INPUT TENSORS (image incoder)
  printf("Preparing input tensor (image incoder)...\n");
  std::vector<Ort::Value> img_encoder_input_tensor;
  img_encoder_input_tensor.push_back(std::move(Ort::Value::CreateTensor<float>(
      memory_info,
      input_images[0].ptr<float>(),
      input_images[0].total(),
      img_encoder_input_nodes[0].dim.data(),
      img_encoder_input_nodes[0].dim.size()))
  );  
  printf("Input tensors ready (image incoder).\n");

  //1.4) RUN INFERENCE (image incoder)
  printf("Inference (image incoder)...\n");
  std::vector<const char*> img_encoder_input_names,img_encoder_output_names;
  for(auto &node:img_encoder_input_nodes)  img_encoder_input_names.push_back(node.name);
  for(auto &node:img_encoder_output_nodes) img_encoder_output_names.push_back(node.name);
  std::vector<Ort::Value> img_encoder_out;

  img_encoder_out = std::move(img_encoder_session->Run(
          Ort::RunOptions{ nullptr },
          img_encoder_input_names.data(),    //input names
          img_encoder_input_tensor.data(),   //input tensor
          img_encoder_input_tensor.size(),   //input tensor size
          img_encoder_output_names.data(),   //output names
          img_encoder_output_names.size())); //output tensors size
 /* 
 img_encoder_out = std::move(img_encoder_session->Run(
          Ort::RunOptions{ nullptr },
          img_encoder_input_names.data(),    //input names
          img_encoder_input_tensor.data(),   //input tensor
          img_encoder_input_tensor.size(),   //input tensor size
          img_encoder_output_names.data(),   //output names
          img_encoder_out.size()));          //output tensors size
  */
  
  //A ryouchinsa ho fa diferent:
  //sessionEncoder->Run(
  //  runOptionsEncoder, //
  //  inputNames.data(),    //input names
  //  &inputTensor,         //input tensor
  //  1,                    //input tensor size
  //  outputNames.data(),   //output names
  //  outputTensors.data(), //output tensors
  //  outputTensors.size());//output tensors size
    
  printf("Inference DONE (image encoder).\n");

  //////////////////////
  // IMAGE DECODER
  /////////////////////
  //input_name= [image_embeddings] ===> [1,256,64,64]
  //input_name= [high_res_features1] ===> [1,32,256,256]
  //input_name= [high_res_features2] ===> [1,64,128,128]
  //input_name= [point_coords] ===> [-1,-1,2]
  //input_name= [point_labels] ===> [-1,-1]
  //input_name= [mask_input] ===> [-1,1,256,256]
  //input_name= [has_mask_input] ===> [-1]
  //input_name= [orig_im_size] ===> [2]
  //output_name= [masks] ==> [-1,-1,-1,-1]
  //output_name= [iou_predictions] ==> [-1,4]
  //output_name= [low_res_masks] ==> [-1,-1,-1,-1]

  //1.1) Create ORT Session (image decoder)
  printf("Creating ORT Session (image decoder)...\n");
  Ort::Env img_decoder_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING,"img_decoder");
  Ort::SessionOptions img_decoder_options = Ort::SessionOptions();
  img_decoder_options.SetIntraOpNumThreads(2);
  Ort::Session* img_decoder_session = new Ort::Session(img_decoder_env, "checkpoints/ryouchinsa/sam2.1_tiny.onnx", img_decoder_options);
  //Ort::Session* img_encoder_session = new Ort::Session(img_encoder_env, "checkpoints/ailia/large/image_encoder_hiera_l.onnx", img_encoder_options);
  printf("ORT session created (image decoder).\n");


  //1.2) Obtain information about the ONNX model (image decoder)
  printf("Obtaining information about the ONNX model (image decoder)...\n");
  std::vector<Node> img_decoder_input_nodes;
  std::vector<Node> img_decoder_output_nodes;
  load_onnx_info(img_decoder_session, img_decoder_input_nodes, img_decoder_output_nodes,"img_decoder");
  printf("Information obtained (image decoder).\n");

  //1.3) PREPARE INPUT TENSORS (image decoder)
  printf("Preparing input tensor (image decoder)...\n");
  std::vector<Ort::Value> img_decoder_input_tensor;

  //embedding and hight_res_feats
  printf("Creating img_decoder_input_tensor (image_embed, high_res_feats_0, high_res_feats_1)\n");
  img_decoder_input_tensor.push_back(std::move(img_encoder_out[0]));    // image_embed
  img_decoder_input_tensor.push_back(std::move(img_encoder_out[1]));    // high_res_feats_0
  img_decoder_input_tensor.push_back(std::move(img_encoder_out[2]));    // high_res_feats_1
  
  /*
  //Format prompt information (bounding box or point)
  cv::Rect box = {745,695,145,230};
  //cv::Point point = {600,450};
  cv::Point point = {600,450};
  box.x = 1024*((float)box.x / ori_img_cols);
  box.y = 1024*((float)box.y / ori_img_rows);
  box.width = 1024*((float)box.width / ori_img_cols);
  box.height = 1024*((float)box.height / ori_img_rows);
  point.x = 1024*((float)point.x / ori_img_cols);
  point.y = 1024*((float)point.y / ori_img_rows);
  int type = 1; // 0=box，1=point
  std::vector<float> point_val, point_labels;
  if(type == 1){ // 0=box，1=point
    point_val = {(float)box.x,(float)box.y,(float)box.x+box.width,(float)box.y+box.height};//xyxy
    point_labels = {2,3};
    img_decoder_input_nodes[0].dim = {1,2,2}; //Change info from the model
    img_decoder_input_nodes[1].dim = {1,2};   //Change info from the model
  }else if(type == 1){
    point_val = {(float)point.x,(float)point.y};//xy
    point_labels = {1};
    img_decoder_input_nodes[0].dim = {1,1,2}; //Change info from the model
    img_decoder_input_nodes[1].dim = {1,1};   //Change info from the model
  }
  std::vector<int64> frame_size = {ori_img_rows,ori_img_cols};
  //std::vector<float> frame_size = {static_cast<float>(ori_img_rows), static_cast<float>(ori_img_cols)};

  printf("Creating img_decoder_input_tensor (point_val)\n");
  printf("\tNumber of elements in the data buffer = %lu\n", point_val.size());
  printf("\tTensor shape dimensions = %lu\n", img_decoder_input_nodes[0].dim.size());
  img_decoder_input_tensor.push_back(Ort::Value::CreateTensor<float>(
        memory_info, 
        point_val.data(), //values
        point_val.size(), //size of values
        img_decoder_input_nodes[0].dim.data(), //shape of values
        img_decoder_input_nodes[0].dim.size())); //shape of values

  printf("Creating img_decoder_input_tensor (point_labels)\n");
  printf("\tNumber of elements in the data buffer = %lu\n", point_labels.size());
  printf("\tTensor shape dimensions = %lu\n", img_decoder_input_nodes[1].dim.size());
  img_decoder_input_tensor.push_back(Ort::Value::CreateTensor<float>(
        memory_info, 
        point_labels.data(), 
        point_labels.size(),
        img_decoder_input_nodes[1].dim.data(),
        img_decoder_input_nodes[1].dim.size()));
  */

  //ryouchinsa style:
  //TODO: Still doing this Aimol style
  /*
  //setDecorderTensorsEmbeddings
  img_decoder_input_tensor.push_back(Ort::Value::CreateTensor<float>(memoryInfo, (float*)outputTensorValuesEncoder.data(), outputTensorValuesEncoder.size(), outputShapeEncoder.data(), outputShapeEncoder.size()));
  img_decoder_input_tensor.push_back(Ort::Value::CreateTensor<float>(memoryInfo, (float*)highResFeatures1.data(), highResFeatures1.size(), highResFeatures1Shape.data(), highResFeatures1Shape.size()));
  img_decoder_input_tensor.push_back(Ort::Value::CreateTensor<float>(memoryInfo, (float*)highResFeatures2.data(), highResFeatures2.size(), highResFeatures2Shape.data(), highResFeatures2Shape.size()));
  */

  //setDecorderTensorsPointsLabels
  std::vector<float> inputPointValues, inputLabelValues;
  inputPointValues.push_back((float)600);
  inputPointValues.push_back((float)450);
  inputLabelValues.push_back((float)0);
  int numPoints = (int)inputLabelValues.size();
  int batchNum = 1;
  std::vector<int64_t> inputPointShape = {batchNum, numPoints, 2};
  std::vector<int64_t> inputLabelShape = {batchNum, numPoints};
  img_decoder_input_tensor.push_back(Ort::Value::CreateTensor<float>(memory_info, inputPointValues.data(), 2 * numPoints * batchNum, inputPointShape.data(), inputPointShape.size()));
  img_decoder_input_tensor.push_back(Ort::Value::CreateTensor<float>(memory_info, inputLabelValues.data(), numPoints * batchNum, inputLabelShape.data(), inputLabelShape.size()));

  //setDecorderTensorsMaskInput
  const size_t maskInputSize = 256 * 256;
  std::vector<float> previousMaskInputValues;
  float maskInputValues[maskInputSize];
  memset(maskInputValues, 0, sizeof(maskInputValues));
  float hasMaskValues[] = {0};
  std::vector<int64_t> maskInputShape = {1, 1, 256, 256},
  hasMaskInputShape = {1};
  if(hasMaskValues[0] == 1){
    img_decoder_input_tensor.push_back(Ort::Value::CreateTensor<float>(memory_info, previousMaskInputValues.data(), maskInputSize, maskInputShape.data(), maskInputShape.size()));
  }else{
    img_decoder_input_tensor.push_back(Ort::Value::CreateTensor<float>(memory_info, maskInputValues, maskInputSize, maskInputShape.data(), maskInputShape.size()));
  }
  img_decoder_input_tensor.push_back(Ort::Value::CreateTensor<float>(memory_info, hasMaskValues, 1, hasMaskInputShape.data(), hasMaskInputShape.size()));

  //setDecorderTensorsImageSize
  std::vector<int64_t> orig_im_size_values_int64 = {ori_img_rows, ori_img_cols};
  //std::vector<float> orig_im_size_values_float = {(float)inputShapeEncoder[2], (float)inputShapeEncoder[3]};
  std::vector<int64_t> origImSizeShape = {2};
  img_decoder_input_tensor.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, orig_im_size_values_int64.data(), 2, origImSizeShape.data(), origImSizeShape.size()));
  cv::Mat outputMask = cv::Mat((int)orig_im_size_values_int64[0], (int)orig_im_size_values_int64[1], CV_8UC1, cv::Scalar(0));
  
  //1.4) RUN INFERENCE (image decoder)
  printf("Inference (image decoder)...\n");
  std::vector<const char*> img_decoder_input_names, img_decoder_output_names;
  for(auto &node:img_decoder_input_nodes)  img_decoder_input_names.push_back(node.name);
  for(auto &node:img_decoder_output_nodes) img_decoder_output_names.push_back(node.name);
  std::vector<Ort::Value> img_decoder_out;
  img_decoder_out = std::move(img_decoder_session->Run(
          Ort::RunOptions{ nullptr },
          img_decoder_input_names.data(),
          img_decoder_input_tensor.data(),
          img_decoder_input_tensor.size(), 
          img_decoder_output_names.data(), 
          img_decoder_output_names.size())); 
  printf("Inference DONE (image decoder).\n");

  /////////////////
  // POSTPROCESS
  /////////////////

  //look for tensor by name
  int tensor_pos = tensorByName(img_decoder_out, img_decoder_output_names, "masks");
  printf("tensor %s found at pos %d\n", "masks", tensor_pos);

  tensor_pos = tensorByName(img_decoder_out, img_decoder_output_names, "iou_predictions");
  printf("tensor %s found at pos %d\n", "iou_predictions", tensor_pos);


  //std::vector<Ort::Value> output_tensors;
  //output_tensors.push_back(std::move(img_decoder_out[2])); //pred_mask
  printf("Postprocessing...\n");

  //aimol
  /*
  cv::Rect box = {745,695,145,230};
  cv::Point point = {600,450};
  std::vector<Ort::Value> output_tensors;
  output_tensors.push_back(std::move(img_decoder_out[2])); //pred_mask 
  printf("Postprocessing...\n");
  postprocess(output_tensors, &image, box, point);
  //postprocess2(output_tensors, &image, box, point);
  cv::imshow("Image", image);
  cv::waitKey(0);
  */
  
  
  int maxScoreIdx = 0;
  float maxScore = 0;
  auto scoreShape = img_decoder_out[1].GetTensorTypeAndShapeInfo().GetShape();
  auto scoreValues = img_decoder_out[1].GetTensorMutableData<float>();
  int scoreNum = (int)scoreShape[1];
  for(int i = 0; i < scoreNum; i++){
    if(scoreValues[i] > maxScore){
      maxScore = scoreValues[i];
      maxScoreIdx = i;
    }
  }
  
  int offsetMask = maxScoreIdx * outputMask.rows * outputMask.cols;
  int offsetLowRes = maxScoreIdx * maskInputSize;
  auto maskValues = img_decoder_out[0].GetTensorMutableData<float>();
  for (int i = 0; i < outputMask.rows; i++) {
    for (int j = 0; j < outputMask.cols; j++) {
        outputMask.at<uchar>(i, j) = maskValues[offsetMask + i * outputMask.cols + j] > 0 ? 255 : 0;
    }
  }

  
  cv::imshow("Image", outputMask);
  cv::waitKey(0);
  

  printf("Postprocessing DONE.\n");
  
  return 0;
}
