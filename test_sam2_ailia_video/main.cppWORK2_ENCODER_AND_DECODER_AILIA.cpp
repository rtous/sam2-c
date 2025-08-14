#include <stdio.h>  
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp> 
#include <opencv2/core/utils/filesystem.hpp>
#include <print>
#include <iostream> 
#include "onnxruntime_utils.h"

//NOTE: The data() function returns a pointer to the block of memory where a vector's elements are stored.

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

int main()
{
  int frame_number = 0;

  // 1) Create CPU
  printf("Allocating memory...\n");
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  // MODELS
  /*
    string models_dir = "checkpoints/ailia/v2_1/tiny";
    image_encoder_ = std::make_unique<OrtModel>(env_, join_path(models_dir, "image_encoder.onnx"));
    prompt_encoder_ = std::make_unique<OrtModel>(env_, join_path(models_dir, "prompt_encoder_hiera.onnx"));
    mask_decoder_ = std::make_unique<OrtModel>(env_, join_path(models_dir, "mask_decoder.onnx"));

    memory_encoder_ = std::make_unique<OrtModel>(env_, join_path(models_dir, "memory_encoder.onnx"));
    memory_attention_ = std::make_unique<OrtModel>(env_, join_path(models_dir, "memory_attention.onnx"));
    mlp_hiera_ = std::make_unique<OrtModel>(env_, join_path(models_dir, "mlp_hiera.onnx"));
    obj_ptr_ = std::make_unique<OrtModel>(env_, join_path(models_dir, "obj_ptr_tpos_proj_hiera.onnx"));
    */
    

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
  
  /////////////////
  // IMAGE ENCODER
  /////////////////
  //current = ailia (works with ailia and with ryouchinsa)
  //ailia:
  //input_name= [input_image] ===> [1,3,1024,1024]
  //output_name= [vision_features] ==> [1,256,64,64]
  //output_name= [vision_pos_enc_0] ==> [1,256,256,256]
  //output_name= [vision_pos_enc_1] ==> [1,256,128,128]
  //output_name= [vision_pos_enc_2] ==> [1,256,64,64]
  //output_name= [backbone_fpn_0] ==> [1,32,256,256]
  //output_name= [backbone_fpn_1] ==> [1,64,128,128]
  //output_name= [backbone_fpn_2] ==> [1,256,64,64]
  //vs ryouchinsa:
  //input_name= [input] ===> [1,3,1024,1024]
  //output_name= [image_embeddings] ==> [1,256,64,64]
  //output_name= [high_res_features1] ==> [1,32,256,256]
  //output_name= [high_res_features2] ==> [1,64,128,128]

  //1) CREATE MODEL (image encoder)
  printf("/*********************************/\n");
  printf("Creating model (image encoder)...\n");
  OrtModel img_encoder = OrtModel("img_encoder", "checkpoints/ailia/v2_1/tiny/image_encoder_hiera_t_2.1.onnx");
  

  //2) PREPARE INPUT TENSORS (image encoder)
  printf("Preparing input tensors (image encoder)...\n");
  std::vector<Ort::Value> img_encoder_input_tensor;
  img_encoder_input_tensor.push_back(std::move(Ort::Value::CreateTensor<float>(
      memory_info,
      input_images[img_encoder.inputIdxByName("input_image")].ptr<float>(),
      input_images[img_encoder.inputIdxByName("input_image")].total(),
      img_encoder.inputs[img_encoder.inputIdxByName("input_image")].shape.data(),
      img_encoder.inputs[img_encoder.inputIdxByName("input_image")].shape.size()))
  );

  //3) RUN INFERENCE (image encoder)
  printf("Inference (image encoder)...\n");
  std::vector<Ort::Value> img_encoder_out  = img_encoder.run(img_encoder_input_tensor);
  
  //////////////////////
  // PROMPT ENCODER 
  /////////////////////
  //alia (no in ryouchinsa, integrated in the decoder)
  //input_name= [coords] ===> [-1,-1,2] ===> float32
  //input_name= [labels] ===> [-1,-1] ===> int32
  //input_name= [masks] ===> [-1,-1,-1] ===> float32
  //input_name= [masks_enable] ===> [1] ===> int32
  //output_name= [sparse_embeddings] ===> [-1,-1,256] ===> float32
  //output_name= [dense_embeddings] ===> [-1,256,-1,-1] ===> float32
  //output_name= [dense_pe] ===> [1,256,64,64] ===> float32


  //1) CREATE MODEL (prompt encoder)
  printf("/*********************************/\n");
  printf("Creating model (prompt encoder)...\n");
  OrtModel prompt_encoder = OrtModel("prompt_encoder", "checkpoints/ailia/v2_1/tiny/prompt_encoder_hiera_t_2.1.onnx");
  
  //2) PREPARE INPUT TENSORS (prompt encoder)
  printf("Preparing input tensors (prompt encoder)...\n");
  std::vector<Ort::Value> prompt_encoder_input_tensor;

  //coords and labels
  std::vector<float> inputPointValues;
  std::vector<int> inputLabelValues;
  inputPointValues.push_back((float)600);
  inputPointValues.push_back((float)450);
  inputLabelValues.push_back(0);
  int numPoints = (int)inputLabelValues.size();
  int batchNum = 1;
  std::vector<int64_t> inputPointShape = {batchNum, numPoints, 2};
  std::vector<int64_t> inputLabelShape = {batchNum, numPoints};
  prompt_encoder_input_tensor.push_back(Ort::Value::CreateTensor<float>(memory_info, inputPointValues.data(), 2 * numPoints * batchNum, inputPointShape.data(), inputPointShape.size()));
  prompt_encoder_input_tensor.push_back(Ort::Value::CreateTensor<int32_t>(memory_info, inputLabelValues.data(), numPoints * batchNum, inputLabelShape.data(), inputLabelShape.size()));

  //masks
  const size_t maskInputSize = 256 * 256;
  std::vector<float> previousMaskInputValues;
  float maskInputValues[maskInputSize];
  memset(maskInputValues, 0, sizeof(maskInputValues));
  int hasMaskValues[] = {0};
  std::vector<int64_t> maskInputShape = {1, 256, 256},
  hasMaskInputShape = {1};
  if(hasMaskValues[0] == 1){
    prompt_encoder_input_tensor.push_back(Ort::Value::CreateTensor<float>(memory_info, previousMaskInputValues.data(), maskInputSize, maskInputShape.data(), maskInputShape.size()));
  } else{
      prompt_encoder_input_tensor.push_back(Ort::Value::CreateTensor<float>(memory_info, maskInputValues, maskInputSize, maskInputShape.data(), maskInputShape.size()));
  }

  //masks_enable
  prompt_encoder_input_tensor.push_back(Ort::Value::CreateTensor<int32_t>(memory_info, hasMaskValues, 1, hasMaskInputShape.data(), hasMaskInputShape.size()));

  //3) RUN INFERENCE (prompt encoder)
  printf("Inference (prompt encoder)...\n");
  std::vector<Ort::Value> prompt_encoder_out  = prompt_encoder.run(prompt_encoder_input_tensor);
  
  //////////////////////
  // IMAGE DECODER
  /////////////////////
  //current = ryouchinsa (works with ryouchinsa)
  //ailia:
  //input_name= [image_embeddings] ===> [1,256,64,64] <- image_encoder[vision_features]
  //input_name= [image_pe] ===> [1,256,64,64]
  //input_name= [sparse_prompt_embeddings] ===> [1,-1,256]
  //input_name= [dense_prompt_embeddings] ===> [1,256,64,64] 
  //input_name= [high_res_features1] ===> [1,32,256,256] <- image_encoder[backbone_fpn_0]
  //input_name= [high_res_features2] ===> [1,64,128,128] <- image_encoder[backbone_fpn_1]
  //output_name= [masks] ===> [1,-1,256,256]
  //output_name= [iou_pred] ===> [-1,4]
  //output_name= [sam_tokens_out] ===> [-1,-1,256]
  //output_name= [object_score_logits] ===> [-1,1]
  //vs ryouchinsa:
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

  //1) CREATE MODEL (image decoder)
  printf("/*********************************/\n");
  printf("Creating model (image decoder)...\n");
  //OrtModel img_decoder = OrtModel("img_decoder", "checkpoints/ryouchinsa/sam2.1_tiny.onnx"); 
  OrtModel img_decoder = OrtModel("img_decoder", "checkpoints/ailia/v2_1/tiny/mask_decoder_hiera_t_2.1.onnx");
  
  //2) PREPARE INPUT TENSORS (image decoder)
  printf("Preparing input tensor (image decoder)...\n");
  std::vector<Ort::Value> img_decoder_input_tensor;
  
  //[image_embeddings]
  int img_encoder_out_vision_features_idx = img_encoder.outputIdxByName("vision_features");
  img_decoder_input_tensor.push_back(std::move(img_encoder_out[img_encoder_out_vision_features_idx]));    // image_embed
  
  //[image_pe]
  int prompt_encoder_out_dense_pe_idx = prompt_encoder.outputIdxByName("dense_pe");
  img_decoder_input_tensor.push_back(std::move(prompt_encoder_out[prompt_encoder_out_dense_pe_idx]));    // image_embed
  
  //[sparse_prompt_embeddings]
  int prompt_encoder_out_sparse_embeddings_idx = prompt_encoder.outputIdxByName("sparse_embeddings");
  img_decoder_input_tensor.push_back(std::move(prompt_encoder_out[prompt_encoder_out_sparse_embeddings_idx]));    // image_embed

  //[dense_prompt_embeddings]
  int prompt_encoder_out_dense_embeddings_idx = prompt_encoder.outputIdxByName("dense_embeddings");
  img_decoder_input_tensor.push_back(std::move(prompt_encoder_out[prompt_encoder_out_dense_embeddings_idx]));    // image_embed

  //[high_res_features1]
  int img_encoder_out_high_res_features1_idx = img_encoder.outputIdxByName("backbone_fpn_0");
  img_decoder_input_tensor.push_back(std::move(img_encoder_out[img_encoder_out_high_res_features1_idx]));    // high_res_features1
  
  //[high_res_features2]
  int img_encoder_out_high_res_features2_idx = img_encoder.outputIdxByName("backbone_fpn_1"); 
  img_decoder_input_tensor.push_back(std::move(img_encoder_out[img_encoder_out_high_res_features2_idx]));    // high_res_features2

  //3) RUN INFERENCE (image decoder)
  printf("Inference (image decoder)...\n");
  std::vector<Ort::Value> img_decoder_out  = img_decoder.run(img_decoder_input_tensor);

  /////////////////
  // POSTPROCESS
  /////////////////
  printf("Postprocessing...\n");

  int img_decoder_out_iou_predictions_idx = img_decoder.outputIdxByName("iou_pred"); 
  
  int maxScoreIdx = 0;
  float maxScore = 0;
  auto scoreShape = img_decoder_out[img_decoder_out_iou_predictions_idx].GetTensorTypeAndShapeInfo().GetShape();
  auto scoreValues = img_decoder_out[img_decoder_out_iou_predictions_idx].GetTensorMutableData<float>();
  int scoreNum = (int)scoreShape[1];
  for(int i = 0; i < scoreNum; i++){
    printf("Checking mask scores number %d = %f\n", i, maxScore);
    if(scoreValues[i] > maxScore){
      maxScore = scoreValues[i];
      maxScoreIdx = i;
    }
  }

  int img_decoder_out_masks_idx = img_decoder.outputIdxByName("masks");
  float* maskValues = img_decoder_out[img_decoder_out_masks_idx].GetTensorMutableData<float>();
  
  //original image size (ori_img_cols = image.size[1], ori_img_rows = image.size[0])
  std::vector<int64_t> orig_im_size_values_int64 = {ori_img_rows, ori_img_cols};
  
  /*
  //assume maskInputSize = 256 * 256;
  cv::Mat outputMask = cv::Mat((int)orig_im_size_values_int64[0], (int)orig_im_size_values_int64[1], CV_8UC1, cv::Scalar(0));
  int offsetMask = maxScoreIdx * outputMask.rows * outputMask.cols;
  //int offsetLowRes = maxScoreIdx * maskInputSize;
  for (int i = 0; i < outputMask.rows; i++) {
    for (int j = 0; j < outputMask.cols; j++) {
        printf("maskValues[offsetMask + i * outputMask.cols + j] = %f\n", maskValues[offsetMask + i * outputMask.cols + j]);
        outputMask.at<uchar>(i, j) = maskValues[offsetMask + i * outputMask.cols + j] > 0 ? 255 : 0;
    }
  }
  cv::imshow("Image", outputMask);
  cv::waitKey(0);
  */

  //Convert the mask to opencv
  cv::Mat outimg(256, 256, CV_32FC1, maskValues);
  cv::Mat dst;
  outimg.convertTo(dst, CV_8UC1, 255);
  cv::threshold(dst,dst,0,255,cv::THRESH_BINARY);
  cv::imshow("Image", dst);
  cv::waitKey(0);

  printf("Postprocessing DONE.\n");

  return 0;
}
