#include <stdio.h>  
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp> 
#include <opencv2/core/utils/filesystem.hpp>
#include <print>
#include <iostream> 
#include "onnxruntime_utils.h" 

#include <random>
#include <algorithm>  // for std::clamp


//#include "xarray.hpp"
//#include "xtensor/xio.hpp"

//NOTE: The data() function returns a pointer to the block of memory where a vector's elements are stored.

/*
->ONNXpreprocess(encoder)->append_image()-> first frame YES -> annotate_frame() -> process_frame()
                                                                                |
                                                         NO ---------------------

process_frame() = 
image_encoder + memory_banck 
  -> memory_attention + prompt_encoder
    -> image_decoder 
      -> memory_encoder 
        -> memory_bank

NOTE: annotate_frame() - Sets up initial object tracking with user prompts
*/

//TODO
/*
cv::Mat interpolate(const cv::Mat& low_res_multimasks, cv::Size image_size) {
    // low_res_multimasks: shape [B, C, H, W] stored as a 4D cv::Mat
    // image_size: target size (width, height)
    
    int B = low_res_multimasks.size[0];
    int C = low_res_multimasks.size[1];
    int H = image_size.height;
    int W = image_size.width;

    int dims[4] = {B, C, H, W};
    cv::Mat high_res_multimasks(4, dims, CV_32F, cv::Scalar(0));

    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            // Extract [H_low, W_low] single-channel slice
            cv::Mat low_res_slice = low_res_multimasks(cv::Range(b, b+1), cv::Range(c, c+1))
                .reshape(1, low_res_multimasks.size[2]); // now [H_low, W_low]

            cv::Mat high_res_slice;
            cv::resize(low_res_slice, high_res_slice, image_size, 0, 0, cv::INTER_LINEAR);

            // Copy back into destination
            high_res_slice.copyTo(
                high_res_multimasks(cv::Range(b, b+1), cv::Range(c, c+1))
                    .reshape(1, H) // shape [H, W]
            );
        }
    }

    return high_res_multimasks;
}
*/

std::vector<float> trunc_normal(
    const std::vector<size_t>& shape,
    float std = 0.02f,
    float a = -2.0f,
    float b = 2.0f
) {
    // Compute total number of elements
    size_t total_size = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());

    std::cout << "trunc_normal with shape = { ";
    for (size_t s : shape) std::cout << s << " ";
    std::cout << "} → total size = " << total_size << "\n";

    std::vector<float> data;
    data.reserve(total_size);

    // RNG setup
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std);

    float lower = a * std;
    float upper = b * std;

    // Fill vector
    for (size_t i = 0; i < total_size; ++i) {
        float val = dist(gen);
        data.push_back(std::clamp(val, lower, upper));
    }

    return data;
}

class InferenceState {
public:
    TensorCopy prompt_encoder_out_dense_pe;
    TensorCopy prompt_encoder_out_sparse_embeddings;
    TensorCopy prompt_encoder_out_dense_embeddings;
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

void inference_frame(cv::Mat image, 
                    int frame_num,
                    Ort::MemoryInfo &memory_info,
                    OrtModel &img_encoder, 
                    OrtModel &prompt_encoder, 
                    OrtModel &img_decoder, 
                    OrtModel &mem_encoder, 
                    OrtModel &mem_attention,
                    OrtModel &mlp,
                    OrtModel &obj_ptr_tpos_proj_hiera,
                    InferenceState &inference_state)
{
  //int frame_number = 0;

  // 1) Create CPU
  //printf("Allocating memory...\n");
  //auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  // 2) Read input image (or input video)
  //printf("Reading input image...\n");
  
  //cv::Mat image = cv::imread("band.jpg", cv::IMREAD_COLOR);
  //if(image.empty())
  //{
  //  printf("ERROR: Cannot read the input image. \n");
  //  exit(-1);
  //}
  int ori_img_cols = image.size[1];
  int ori_img_rows = image.size[0];
  //printf("Successfully read (cols=%d, rows=%d)...\n", ori_img_cols, ori_img_rows);

  //3) preprocess input image (opencv to onnx format)
  printf("Preprocessing input image...\n");
  std::vector<cv::Mat> input_images;
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
  //input_name= [image] ===> [1,3,1024,1024]
  //output_name= [pix_feat] ==> [1,256,64,64]
  //output_name= [high_res_feat0] ==> [1,32,256,256]
  //output_name= [high_res_feat1] ==> [1,64,128,128]
  //output_name= [vision_feats] ==> [1,256,64,64]
  //output_name= [vision_pos_embed] ==> [4096,1,256]

  //1) CREATE MODEL (image encoder)
  printf("/********** image encoder ***********/\n");
  //printf("Creating model (image encoder)...\n");
  //OrtModel img_encoder = OrtModel("img_encoder", "checkpoints/ailia/v2_1/tiny/image_encoder_hiera_t_2.1.onnx");
  

  //2) PREPARE INPUT TENSORS (image encoder)
  printf("Preparing input tensors (image encoder)...\n");
  std::vector<Ort::Value> img_encoder_input_tensor;
  img_encoder_input_tensor.push_back(std::move(Ort::Value::CreateTensor<float>(
      memory_info,
      input_images[0].ptr<float>(),
      input_images[0].total(),
      img_encoder.inputs[img_encoder.inputIdxByName("input_image")].shape.data(),
      img_encoder.inputs[img_encoder.inputIdxByName("input_image")].shape.size()))
  );

  //3) RUN INFERENCE (image encoder)
  printf("Inference (image encoder)...\n");
  std::vector<Ort::Value> img_encoder_out  = img_encoder.run(img_encoder_input_tensor);

  //4) Save tensors that need to be reuse more than once:
  //[vision_features]
  int img_encoder_out_vision_features_idx = img_encoder.outputIdxByName("vision_features");
  TensorCopy img_encoder_out_vision_features = setTensorCopy(std::move(img_encoder_out[img_encoder_out_vision_features_idx]));  
  
  
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

  if (frame_num == 0)
  {
    //1) CREATE MODEL (prompt encoder)
    printf("/*********** prompt encoder ***********/\n");
    //printf("Creating model (prompt encoder)...\n");
    //OrtModel prompt_encoder = OrtModel("prompt_encoder", "checkpoints/ailia/v2_1/tiny/prompt_encoder_hiera_t_2.1.onnx");
    
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
    if(hasMaskValues[0] == 1)
    {
      prompt_encoder_input_tensor.push_back(Ort::Value::CreateTensor<float>(memory_info, previousMaskInputValues.data(), maskInputSize, maskInputShape.data(), maskInputShape.size()));
    } else{
      prompt_encoder_input_tensor.push_back(Ort::Value::CreateTensor<float>(memory_info, maskInputValues, maskInputSize, maskInputShape.data(), maskInputShape.size()));
    }

    //masks_enable
    prompt_encoder_input_tensor.push_back(Ort::Value::CreateTensor<int32_t>(memory_info, hasMaskValues, 1, hasMaskInputShape.data(), hasMaskInputShape.size()));

    //3) RUN INFERENCE (prompt encoder)
    printf("Inference (prompt encoder)...\n");
    std::vector<Ort::Value> prompt_encoder_out  = prompt_encoder.run(prompt_encoder_input_tensor);

    //Only need to run this for the first frame, so storing copies of the outputs:
    //[dense_pe]
    int prompt_encoder_out_dense_pe_idx = prompt_encoder.outputIdxByName("dense_pe");
    inference_state.prompt_encoder_out_dense_pe = setTensorCopy(std::move(prompt_encoder_out[prompt_encoder_out_dense_pe_idx]));
    
    //[sparse_prompt_embeddings]
    int prompt_encoder_out_sparse_embeddings_idx = prompt_encoder.outputIdxByName("sparse_embeddings");
    inference_state.prompt_encoder_out_sparse_embeddings = setTensorCopy(std::move(prompt_encoder_out[prompt_encoder_out_sparse_embeddings_idx]));
    
    //[dense_prompt_embeddings]
    int prompt_encoder_out_dense_embeddings_idx = prompt_encoder.outputIdxByName("dense_embeddings");
    inference_state.prompt_encoder_out_dense_embeddings = setTensorCopy(std::move(prompt_encoder_out[prompt_encoder_out_dense_embeddings_idx]));
  }

  //////////////////////
  // MEM ATTENTION
  /////////////////////
  //alia 
  //input_name= [curr] ===> [4096,1,256] ===> float32
  //input_name= [memory_1] ===> [-1,1,64] ===> float32
  //input_name= [memory_2] ===> [-1,1,64] ===> float32
  //input_name= [curr_pos] ===> [4096,1,256] ===> float32
  //input_name= [memory_pos_1] ===> [-1,1,64] ===> float32
  //input_name= [memory_pos_2] ===> [-1,1,64] ===> float32
  //input_name= [attention_mask_1] ===> [-1,1] ===> bool
  //input_name= [attention_mask_2] ===> [-1,1] ===> bool
  //output_name= [pix_feat] ===> [4096,1,256] ===> float32
  //vs ryouchinsa:
  //input_name= [current_vision_feat] ===> [1,256,64,64]
  //input_name= [current_vision_pos_embed] ===> [4096,1,256]
  //input_name= [memory_0] ===> [-1,256]
  //input_name= [memory_1] ===> [-1,64,64,64]
  //input_name= [memory_pos_embed] ===> [-1,1,64]
  //output_name= [image_embed] ==> [1,256,64,64]

  if (frame_num > 0)
  {
      //1) CREATE MODEL (mem_attention)
      printf("/************ mem_attention *************/\n");
      //printf("Creating model (mem_attention encoder)...\n");
      //OrtModel mem_attention = OrtModel("mem_attention", "checkpoints/ailia/v2_1/tiny/memory_attention_hiera_t_2.1.opt.onnx");
      
      //2) PREPARE INPUT TENSORS (mem_attention)
      printf("Preparing input tensors (mem_attention)...\n");
      std::vector<Ort::Value> mem_attention_input_tensor;

      /*ryouchinsa:
        //img_encoder_out[3] in ryouchinsa ([high_res_feat1]->[current_vision_feat] [1,256,64,64])
        //img_encoder_out[4] in ryouchinsa ([vision_pos_embed]->[current_vision_pos_embed] [4096,1,256])
        //memory_1 in ryouchinsa ([memory_0]) Hi posa la màscara del frame primer
        //memory_2 in ryouchinsa ([memory_1]) Hi posa la màscara del frame actual
        //memory_pos_1 in ryouchinsa ([memory_pos_embed]) Ho calcula en base les tres sortides del mem_encoder
        //memory_pos_2 in ryouchinsa???? 
      */

      //From image incoder ouputs:
      //output_name= [vision_features] ==> [1,256,64,64]
      //output_name= [vision_pos_enc_0] ==> [1,256,256,256]
      //output_name= [vision_pos_enc_1] ==> [1,256,128,128]
      //output_name= [vision_pos_enc_2] ==> [1,256,64,64]
      //output_name= [backbone_fpn_0] ==> [1,32,256,256]
      //output_name= [backbone_fpn_1] ==> [1,64,128,128]
      //output_name= [backbone_fpn_2] ==> [1,256,64,64]

      //[curr] ===> [4096,1,256] ===> float32
      //img_decoder_input_tensor.push_back(getTensorCopy(img_encoder_out_vision_features, memory_info));
  
      //[memory_1] ===> [-1,1,64] ===> float32

      //[memory_2] ===> [-1,1,64] ===> float32

      //[curr_pos] ===> [4096,1,256] ===> float32

      //[memory_pos_1] ===> [-1,1,64] ===> float32

      //[memory_pos_2] ===> [-1,1,64] ===> float32

      //[attention_mask_1] ===> [-1,1] ===> bool

      //[attention_mask_2] ===> [-1,1] ===> bool

      //3) RUN INFERENCE (mem_attention encoder)
      //printf("Inference (mem_attention)...\n");
      //std::vector<Ort::Value> mem_attention_out  = mem_attention.run(mem_attention_input_tensor);

  } else {
    //TODO: What happens with the first frame???
    //mem_attention_out.push_back(std::move(img_encoder_out[3]));
  }


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
  printf("/*********** image decoder ************/\n");
  //printf("Creating model (image decoder)...\n");
  //OrtModel img_decoder = OrtModel("img_decoder", "checkpoints/ryouchinsa/sam2.1_tiny.onnx"); 
  //OrtModel img_decoder = OrtModel("img_decoder", "checkpoints/ailia/v2_1/tiny/mask_decoder_hiera_t_2.1.onnx");
  
  //2) PREPARE INPUT TENSORS (image decoder)
  printf("Preparing input tensor (image decoder)...\n");
  std::vector<Ort::Value> img_decoder_input_tensor;
  
  //[vision_features]
  //work with a copy as we will use again in memory_encoder
  //int img_encoder_out_vision_features_idx = img_encoder.outputIdxByName("vision_features");
  //img_decoder_input_tensor.push_back(std::move(img_encoder_out[img_encoder_out_vision_features_idx]));    // vision_features
  //TensorCopy img_encoder_out_vision_features = setTensorCopy(std::move(img_encoder_out[img_encoder_out_vision_features_idx]));  
  img_decoder_input_tensor.push_back(getTensorCopy(img_encoder_out_vision_features, memory_info));
  
  //[image_pe]
  int prompt_encoder_out_dense_pe_idx = prompt_encoder.outputIdxByName("dense_pe");
  //img_decoder_input_tensor.push_back(std::move(prompt_encoder_out[prompt_encoder_out_dense_pe_idx]));    // image_embed
  img_decoder_input_tensor.push_back(getTensorCopy(inference_state.prompt_encoder_out_dense_pe, memory_info));

  //[sparse_prompt_embeddings]
  int prompt_encoder_out_sparse_embeddings_idx = prompt_encoder.outputIdxByName("sparse_embeddings");
  //img_decoder_input_tensor.push_back(std::move(prompt_encoder_out[prompt_encoder_out_sparse_embeddings_idx]));    // image_embed
  img_decoder_input_tensor.push_back(getTensorCopy(inference_state.prompt_encoder_out_sparse_embeddings, memory_info));

  //[dense_prompt_embeddings]
  int prompt_encoder_out_dense_embeddings_idx = prompt_encoder.outputIdxByName("dense_embeddings");
  //img_decoder_input_tensor.push_back(std::move(prompt_encoder_out[prompt_encoder_out_dense_embeddings_idx]));    // image_embed
  img_decoder_input_tensor.push_back(getTensorCopy(inference_state.prompt_encoder_out_dense_embeddings, memory_info));

  //[high_res_features1]
  int img_encoder_out_high_res_features1_idx = img_encoder.outputIdxByName("backbone_fpn_0");
  img_decoder_input_tensor.push_back(std::move(img_encoder_out[img_encoder_out_high_res_features1_idx]));    // high_res_features1
  
  //[high_res_features2]
  int img_encoder_out_high_res_features2_idx = img_encoder.outputIdxByName("backbone_fpn_1"); 
  img_decoder_input_tensor.push_back(std::move(img_encoder_out[img_encoder_out_high_res_features2_idx]));    // high_res_features2

  //3) RUN INFERENCE (image decoder)
  printf("Inference (image decoder)...\n");
  std::vector<Ort::Value> img_decoder_out  = img_decoder.run(img_decoder_input_tensor);


  /* ailiol: 
  //Guarda el [masks] del primer i l'actual???
  if(infer_status.current_frame == 0)[[unlikely]]{
        infer_status.obj_ptr_first.push_back(std::move(img_decoder_out[0]));
    }else{
        infer_status.obj_ptr_recent.push(std::move(img_decoder_out[0]));
    }
  */
  if (frame_num == 0)
  {

  }

  //////////////////////
  // MLP
  /////////////////////
  //input_name= [x] ===> [-1,256] ===> float32
  //output_name= [x_out] ===> [-1,256] ===> float32

  //1) CREATE MODEL (mlp) //in the main
  printf("/*********** mlp ************/\n");
  //printf("Creating model (mlp)...\n");
  //OrtModel mlp = OrtModel("mlp", "checkpoints/ailia/v2_1/tiny/mlp_hiera_t_2.1.onnx");
  
  //2) PREPARE INPUT TENSORS (mlp)
  printf("Preparing input tensor (mlp)...\n");
  std::vector<Ort::Value> mlp_input_tensor;
  
  //[x] ===> [-1,256]
  int img_decoder_out_sam_tokens_out_idx = img_decoder.outputIdxByName("sam_tokens_out");
  TensorCopy img_decoder_out_sam_tokens_out = setTensorCopy(std::move(img_decoder_out[img_decoder_out_sam_tokens_out_idx]));
  //num elements = [1024]
  //shape = [[1,4,256]]
  //type = [float32]
  //need reshape [1,4,256] -> [4,256] (I guess, maybe would work with [1,256]?)
  TensorCopy img_decoder_out_sam_tokens_out_first = slice_1xNxC_toNxC(img_decoder_out_sam_tokens_out);
  //TODO: Also works if slice_1xNxC_toNxC, change if problems
  printTensorCopyInfo(img_decoder_out_sam_tokens_out_first);
  mlp_input_tensor.push_back(std::move(getTensorCopy(img_decoder_out_sam_tokens_out_first, memory_info)));    // x

   //3) RUN INFERENCE (mlp)
  printf("Inference (mlp)...\n");
  std::vector<Ort::Value> imlp_out  = mlp.run(mlp_input_tensor);

  //////////////////////
  // MEM ENCODER
  /////////////////////
  //alia 
  //input_name= [pix_feat] ===> [1,256,64,64] ===> float32
  //input_name= [masks] ===> [1,1,1024,1024] ===> float32
  //output_name= [vision_features] ===> [1,64,64,64] ===> float32
  //output_name= [vision_pos_enc] ===> [1,64,64,64] ===> float32
  ///vs ryouchinsa:
  //input_name= [mask_for_mem] ===> [1,1,1024,1024]
  //input_name= [pix_feat] ===> [1,256,64,64]
  //output_name= [maskmem_features] ==> [1,64,64,64]
  //output_name= [maskmem_pos_enc] ==> [4096,1,64]
  //output_name= [temporal_code] ==> [7,1,1,64]

  //
  //
  //

  
  //1) CREATE MODEL (mem_encoder)
  printf("/*********** mem_encoder **********/\n");
  //printf("Creating model (mem_encoder)...\n");
  //OrtModel mem_encoder = OrtModel("mem_encoder", "checkpoints/ailia/v2_1/tiny/memory_encoder_hiera_t_2.1.onnx");
  
  //2) PREPARE INPUT TENSORS (mem_encoder)
  printf("Preparing input tensors (mem_encoder)...\n");
  std::vector<Ort::Value> mem_encoder_input_tensor;

  //[pix_feat] (TODO: this should be the attention output, but in aimol uses this)
  //int img_encoder_out_vision_features_idx = img_encoder.outputIdxByName("vision_features");
  //mem_encoder_input_tensor.push_back(std::move(img_encoder_out[img_encoder_out_vision_features_idx])); 
  mem_encoder_input_tensor.push_back(std::move(getTensorCopy(img_encoder_out_vision_features, memory_info))); 
  
  //[masks] (We work with a copy as we will use img_decoder_out_masks later)
  //the original mask is upscaled (with a function called interpolated)
  int img_decoder_out_masks_idx = img_decoder.outputIdxByName("masks");
  TensorCopy img_decoder_out_masks = setTensorCopy(std::move(img_decoder_out[img_decoder_out_masks_idx]));  
  //mem_encoder_input_tensor.push_back(getTensorCopy(img_decoder_out_masks, memory_info));
 
  float* maskValues_ = getTensorCopy(img_decoder_out_masks, memory_info).GetTensorMutableData<float>();
  cv::Mat low_res_mask_(256, 256, CV_32FC1, maskValues_);
  //cv::Mat low_res_mask; //TODO: Check if this is necessary
  //low_res_mask_.convertTo(low_res_mask, CV_8UC1, 255);
  cv::Mat high_res_mask;
  cv::Size sam2_image_size = cv::Size(1024, 1024);
  cv::resize(low_res_mask_, high_res_mask, sam2_image_size, 0, 0, cv::INTER_LINEAR);
  //need to use the preprocess function to transform into a tensor)
  std::vector<cv::Mat> input_masks;
  preprocess(high_res_mask, input_masks); 
  
  //mem_encoder_input_tensor.push_back(getTensorCopy(img_decoder_out_masks, memory_info));
  mem_encoder_input_tensor.push_back(std::move(Ort::Value::CreateTensor<float>(
      memory_info,
      input_masks[0].ptr<float>(),
      input_masks[0].total(),
      mem_encoder.inputs[mem_encoder.inputIdxByName("masks")].shape.data(),
      mem_encoder.inputs[mem_encoder.inputIdxByName("masks")].shape.size()))
  ); 

  //3) RUN INFERENCE (mem_encoder decoder)
  printf("Inference (mem_encoder)...\n");
  std::vector<Ort::Value> mem_encoder_out  = mem_encoder.run(mem_encoder_input_tensor);
  
  //for frame_idx in temp_frame_inds:
  //output_dict[storage_key][frame_idx] = consolidated_out
  //_add_output_per_object()
  //clear_non_cond_mem_around_input
  //# clear temporary outputs in `temp_output_dict_per_obj`


  /* ryouchinsa */
  //temp.maskmem_features.push_back(std::move(mem_encoder_out[0]));//[maskmem_features] ==> [1,64,64,64]
  //temp.maskmem_pos_enc.push_back(std::move(mem_encoder_out[1]));//[maskmem_pos_enc] ==> [4096,1,64]
  //temp.temporal_code.push_back(std::move(mem_encoder_out[2]));//[temporal_code] ==> [7,1,1,64]

  //////////////////////
  // STORE IN MEMORY
  //////////////////////
  /*
  Inputs

  From memory encoder:
  //output_name= [vision_features] ===> [1,64,64,64]
  //output_name= [vision_pos_enc] ===> [1,64,64,64]

  From MLP:
  [x_out] ===> [-1,256]

  The output should be used from the memory attention:

  //input_name= [curr] ===> [4096,1,256] ===> float32
  //input_name= [memory_1] ===> [-1,1,64] ===> float32
  //input_name= [memory_2] ===> [-1,1,64] ===> float32
  //input_name= [curr_pos] ===> [4096,1,256] ===> float32
  //input_name= [memory_pos_1] ===> [-1,1,64] ===> float32
  //input_name= [memory_pos_2] ===> [-1,1,64] ===> float32
  //input_name= [attention_mask_1] ===> [-1,1] ===> bool
  //input_name= [attention_mask_2] ===> [-1,1] ===> bool
  */

  //just debugging trunc_normal
  std::vector<size_t> shape = {2, 3, 4};  // 3D tensor (can be N-D)
  std::vector<float> values = trunc_normal(shape);

  size_t total_size = values.size();
  for (size_t i = 0; i < total_size; ++i) {
      std::cout << "values[" << i << "] = " << values[i] << "\n";
  }

  /*First frame:
  def trunc_normal(size, std=0.02, a=-2, b=2):
    values = np.random.normal(loc=0., scale=std, size=size)
    values = np.clip(values, a*std, b*std)
    return values.astype(np.float32)

  self.no_mem_embed = trunc_normal((1, 1, self.hidden_dim), std=0.02)
  self.no_mem_pos_enc = trunc_normal((1, 1, self.hidden_dim), std=0.02)
  # for initial conditioning frames, encode them without using any previous memory
    if self.directly_add_no_mem_embed:
        # directly add no-mem embedding (instead of using the transformer encoder)
        pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
        pix_feat_with_mem = np.transpose(pix_feat_with_mem, (1, 2, 0)).reshape(B, C, H, W)
        return pix_feat_with_mem

    # Use a dummy token on the first frame (to avoid empty memory input to tranformer encoder)
    to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
    to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]


    # Step 2: Concatenate the memories and forward through the transformer encoder
    memory = np.concatenate(to_cat_memory, axis=0)
    memory_pos_embed = np.concatenate(to_cat_memory_pos_embed, axis=0)

    num_obj_ptr_tokens_numpy = np.array((num_obj_ptr_tokens)).astype(np.int64)
    if self.debug:
        print("curr", np.sum(current_vision_feats[0]))
        print("memory", np.sum(memory))
        print("curr_pos", np.sum(current_vision_pos_embeds[0]))
        print("memory_pos", np.sum(memory_pos_embed))
        print("num_obj_ptr_tokens", np.sum(num_obj_ptr_tokens_numpy))
    if self.benchmark:
        start = int(round(time.time() * 1000))

    if self.version == "2.1":
        memory_1 = memory[:-num_obj_ptr_tokens,:,:]
        memory_2 = memory[-num_obj_ptr_tokens:,:,:]
        memory_pos_embed_1 = memory_pos_embed[:-num_obj_ptr_tokens,:,:]
        memory_pos_embed_2 = memory_pos_embed[-num_obj_ptr_tokens:,:,:]
        attention_mask_1 = np.zeros((memory_1.shape[0], memory_1.shape[1]), dtype=np.bool_)
        attention_mask_2 = np.zeros((memory_2.shape[0], memory_2.shape[1]), dtype=np.bool_)
        attention_mask_1[:memory_1.shape[0],:] = True
        attention_mask_2[:memory_2.shape[0],:] = True
  */




  //Coses barrejades:
  //NOTE: conditioned frames: els que han rebut clicks?
  //num_maskmem = 7,  # default 1 input frame + 6 previous frames
  //temp_output_dict_per_obj (consolidate per-object temporary outputs)
  //consolidated_frame_inds (indices of those frames already consolidated)

  //First frame:



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

  //As we use img_decoder_out_masks twice (before in the memory encoder) we need a copy. 
  //float* maskValues = img_decoder_out[img_decoder_out_masks_idx].GetTensorMutableData<float>();
  float* maskValues = getTensorCopy(img_decoder_out_masks, memory_info).GetTensorMutableData<float>();

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
}

int main()
{

  // 1) Create CPU
  printf("Allocating memory...\n");
 Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  
  printf("Creating models...\n");
  OrtModel img_encoder = OrtModel("img_encoder", "checkpoints/ailia/v2_1/tiny/image_encoder_hiera_t_2.1.onnx");
  OrtModel prompt_encoder = OrtModel("prompt_encoder", "checkpoints/ailia/v2_1/tiny/prompt_encoder_hiera_t_2.1.onnx");
  OrtModel img_decoder = OrtModel("img_decoder", "checkpoints/ailia/v2_1/tiny/mask_decoder_hiera_t_2.1.onnx");
  OrtModel mlp = OrtModel("mlp_hiera", "checkpoints/ailia/v2_1/tiny/mlp_hiera_t_2.1.onnx");
  OrtModel mem_encoder = OrtModel("mem_encoder", "checkpoints/ailia/v2_1/tiny/memory_encoder_hiera_t_2.1.onnx");
  OrtModel mem_attention = OrtModel("mem_attention", "checkpoints/ailia/v2_1/tiny/memory_attention_hiera_t_2.1.opt.onnx");
  OrtModel obj_ptr_tpos_proj_hiera = OrtModel("obj_ptr_tpos_proj_hiera", "checkpoints/ailia/v2_1/tiny/obj_ptr_tpos_proj_hiera_t_2.1.onnx");
  
  printf("Initializing inference state...\n");
  InferenceState inference_state = InferenceState();

  printf("Reading video...\n");
  std::string video_path = "footage.mp4";
  cv::VideoCapture capture(video_path);
  if (!capture.isOpened()) return 0;
  cv::Mat frame;
  int i = 0;
  while (true) {
      printf("Frame 1...\n");
      if (!capture.read(frame) || frame.empty()) break;
      printf("Inferencing frame...\n");
      //auto result = sam2->inference(frame);
      inference_frame(frame, i, memory_info, img_encoder, prompt_encoder, img_decoder, mem_encoder, mem_attention, mlp, obj_ptr_tpos_proj_hiera, inference_state);
      cv::imshow("frame", frame);
      cv::waitKey(0);
      i++;
      printf("Frame 1 DONE.\n");
  }
  capture.release();
}