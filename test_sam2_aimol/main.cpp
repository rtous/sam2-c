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

/*void postprocess2(std::vector<Ort::Value> &output_tensors, cv::Mat *ori_img, cv::Rect prompt_box, cv::Point prompt_point){
    float* output =  output_tensors[0].GetTensorMutableData<float>();
    cv::Mat outimg(ori_img->size(),CV_32FC1, output);
    cv::imshow("Image", outimg);
    cv::waitKey(0);
}*/

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

  //1.1) Create ORT Session (image incoder)
  printf("Creating ORT Session (image incoder)...\n");
  Ort::Env img_encoder_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING,"img_encoder");
  Ort::SessionOptions img_encoder_options = Ort::SessionOptions();
  img_encoder_options.SetIntraOpNumThreads(2);
  Ort::Session* img_encoder_session = new Ort::Session(img_encoder_env, "checkpoints/aimol/small/image_encoder.onnx", img_encoder_options);
  //Ort::Session* img_encoder_session = new Ort::Session(img_encoder_env, "checkpoints/ailia/large/image_encoder_hiera_l.onnx", img_encoder_options);
  printf("ORT session created (image incoder).\n");


  //1.2) Obtain information about the ONNX model (image encoder)
  printf("Obtaining information about the ONNX model (image encoder)...\n");
  std::vector<Node> img_encoder_input_nodes;
  std::vector<Node> img_encoder_output_nodes;
  load_onnx_info(img_encoder_session, img_encoder_input_nodes, img_encoder_output_nodes,"img_encoder");
  printf("Information obtained (image incoder).\n");
  //input_name= [image] ===> [1,3,1024,1024]
  //output_name= [pix_feat] ==> [1,256,64,64]
  //output_name= [high_res_feat0] ==> [1,32,256,256]
  //output_name= [high_res_feat1] ==> [1,64,128,128]
  //output_name= [vision_feats] ==> [1,256,64,64]
  //output_name= [vision_pos_embed] ==> [4096,1,256]

  //1.3) PREPARE INPUT TENSORS (image incoder)
  printf("Preparing input tensor (image encoder)...\n");
  std::vector<Ort::Value> img_encoder_input_tensor;
  img_encoder_input_tensor.push_back(std::move(Ort::Value::CreateTensor<float>(
      memory_info,
      input_images[0].ptr<float>(),
      input_images[0].total(),
      img_encoder_input_nodes[0].dim.data(),
      img_encoder_input_nodes[0].dim.size()))
  );  
  printf("Input tensors ready (image encoder).\n");

  //1.4) RUN INFERENCE (image encoder)
  printf("Inference (image encoder)...\n");
  std::vector<const char*> img_encoder_input_names,img_encoder_output_names;
  for(auto &node:img_encoder_input_nodes)  img_encoder_input_names.push_back(node.name);
  for(auto &node:img_encoder_output_nodes) img_encoder_output_names.push_back(node.name);
  std::vector<Ort::Value> img_encoder_out;
  img_encoder_out = std::move(img_encoder_session->Run(
          Ort::RunOptions{ nullptr },
          img_encoder_input_names.data(),
          img_encoder_input_tensor.data(),
          img_encoder_input_tensor.size(), 
          img_encoder_output_names.data(), 
          img_encoder_output_names.size())); 
  printf("Inference DONE (image encoder).\n");

  
  //delete img_encoder_session;


  //////////////////////
  // MEMORY ATTENTION
  /////////////////////
  //input_name= [current_vision_feat] ===> [1,256,64,64]
  //input_name= [current_vision_pos_embed] ===> [4096,1,256]
  //input_name= [memory_0] ===> [-1,256]
  //input_name= [memory_1] ===> [-1,64,64,64]
  //input_name= [memory_pos_embed] ===> [-1,1,64]
  //
  //output_name= [image_embed] ==> [1,256,64,64]


  //2.1) Create ORT Session (memory attention)
  printf("Creating ORT Session (memory attention)...\n");
  Ort::Env mem_attention_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING,"mem_attention");
  Ort::SessionOptions mem_attention_options = Ort::SessionOptions();
  mem_attention_options.SetIntraOpNumThreads(2);
  Ort::Session* mem_attention_session = new Ort::Session(mem_attention_env, "checkpoints/aimol/small/memory_attention.onnx", img_encoder_options);
  //Ort::Session* img_encoder_session = new Ort::Session(img_encoder_env, "checkpoints/ailia/large/image_encoder_hiera_l.onnx", img_encoder_options);
  printf("ORT session created (memory attention).\n");

  //2.2) Obtain information about the ONNX model (memory attention)
  printf("Obtaining information about the ONNX model (memory attention)...\n");
  std::vector<Node> mem_attention_input_nodes;
  std::vector<Node> mem_attention_output_nodes;
  load_onnx_info(mem_attention_session, mem_attention_input_nodes, mem_attention_output_nodes,"mem_attention");
  printf("Information obtained (memory attention).\n");

  std::vector<Ort::Value> mem_attention_out;
  if (frame_number > 0) { //not the first frame

        printf("Memory attention... not first frame, need to infer.\n");

        /*
        //2.3) PREPARE INPUT TENSORS (memory attention)
        printf("Preparing input tensor (memory attention)...\n");

        //*******************************************************************************
        //创建输入数据 curr，curr_pos，memory_1，memory_2，memory_pos_1,memory_pos_2
        std::vector<Ort::Value> mem_attention_input_tensor; // 6
        mem_attention_input_tensor.push_back(std::move(img_encoder_out[3])); //current_vision_feat
        mem_attention_input_tensor.push_back(std::move(img_encoder_out[4])); //current_vision_pos_embed

        size_t obj_buffer_size = 1 + infer_status.obj_ptr_recent.size();//1+0,1+1,1+2,...,1+15

        std::vector<int64_t> dimensions_0{(int64_t)obj_buffer_size,256}; // [y,256]
        std::vector<float> obj_ptrs(obj_buffer_size*256); // first+recent // 16*256

        const float* tensor_data = infer_status.obj_ptr_first[0].GetTensorData<float>();
        std::copy_n(tensor_data, 256, std::begin(obj_ptrs));

        for(size_t i = 0;i<infer_status.obj_ptr_recent.size();i++){
            auto& temp_tensor = infer_status.obj_ptr_recent.at(i);
            tensor_data = temp_tensor.GetTensorData<float>();
            std::copy_n(tensor_data, 256, std::begin(obj_ptrs)+256*(i+1));
        }

        auto memory_1 = Ort::Value::CreateTensor<float>(
                        memory_info,
                        obj_ptrs.data(),
                        obj_ptrs.size(),
                        dimensions_0.data(),
                        dimensions_0.size()
                        );

        size_t features_size = infer_status.status_recent.size(); // 1,2,3,...,7
        std::vector<float> maskmem_features_(features_size*64*64*64);
        for(size_t i = 0;i<features_size;i++){
            auto& temp_tensor = this->infer_status.status_recent.at(i).maskmem_features;
            tensor_data = temp_tensor[0].GetTensorData<float>();
            std::copy_n(tensor_data, 64*64*64, std::begin(maskmem_features_)+64*64*64*i);
        }
        std::vector<int64_t> dimensions_1{(int64_t)features_size,64,64,64}; // [x,64,64,64]
        auto memory_2 = Ort::Value::CreateTensor<float>(
                        memory_info,
                        maskmem_features_.data(),
                        maskmem_features_.size(),
                        dimensions_1.data(),
                        dimensions_1.size()
                        );
        mem_attention_input_tensor.push_back(std::move(memory_1));
        mem_attention_input_tensor.push_back(std::move(memory_2));

        //***********************************************************************
        // memory_pos_embed是由两部分组成的。
        auto& temp_time = infer_status.status_recent.at(features_size-1).temporal_code;
        const float* temporal_code_ = temp_time[0].GetTensorData<float>(); // [7,64]
        std::vector<const float*> temporal_code;
        for(int i = 6;i>=0;i--){
            auto temp = temporal_code_+i*64;
            temporal_code.push_back(temp);
        }
        size_t maskmem_buffer_size = infer_status.status_recent.size();
        size_t memory_pos_1_size = (maskmem_buffer_size*4096)*64;
        size_t memory_pos_2_size = std::min(infer_status.current_frame,16)*256;

        std::vector<float> memory_pos_1_(memory_pos_1_size);
        std::vector<float> memory_pos_2_(memory_pos_2_size,0);

        // a[] , b[4096,1,64], c[1,1,64]
        auto tensor_add = [&](float* a,const float* b,const float* c){
            // b+c,结果保存到a
            for(int i =0;i<4096;i++){
                for(int j =0;j<64;j++){
                    a[i*64+j] = b[i*64+j] + c[j];
                }
            }
        };
        // 第一部分：
        for(size_t j = 0;j<maskmem_buffer_size;j++){
            auto& temp_tensor = this->infer_status.status_recent.at(j).maskmem_pos_enc;
            auto sub = temp_tensor[0].GetTensorData<float>();//[4096,1,64]
            float* p = memory_pos_1_.data() + j*4096*64;
            tensor_add(p,sub,temporal_code.at(j)); // [4096,1,64] + [1,1,64] ->[4096,1,64] + [4096,1,64] ->[4096,1,64]
        }
        std::vector<int64_t> dimensions_3{int64_t(maskmem_buffer_size*4096),1,64}; // [z,1,64]
        std::vector<int64_t> dimensions_4{int64_t(4*std::min(infer_status.current_frame,16)),1,64}; // [num,1,64]

        auto memory_pos_1 = Ort::Value::CreateTensor<float>(
                            memory_info,
                            memory_pos_1_.data(),
                            memory_pos_1_.size(),
                            dimensions_3.data(),
                            dimensions_3.size()
                            );
        auto memory_pos_2 = Ort::Value::CreateTensor<float>(
                        memory_info,
                        memory_pos_2_.data(),
                        memory_pos_2_.size(),
                        dimensions_4.data(),
                        dimensions_4.size()
                        );
        mem_attention_input_tensor.push_back(std::move(memory_pos_1));
        mem_attention_input_tensor.push_back(std::move(memory_pos_2));  
        printf("Input tensors ready (memory attention).\n");

        //1.4) RUN INFERENCE (image incoder)
        printf("Inference (memory attention)...\n");
        std::vector<const char*> mem_attention_input_names,mem_attention_output_names;
        for(auto &node:mem_attention_input_nodes)  mem_attention_input_names.push_back(node.name);
        for(auto &node:mem_attention_output_nodes) mem_attention_output_names.push_back(node.name);
        mem_attention_out = std::move(mem_attention_session->Run(
              Ort::RunOptions{ nullptr },
              input_names.data(),
              mem_attention_input_tensor.data(),
              mem_attention_input_tensor.size(), 
              mem_attention_output_names.data(), 
              mem_attention_output_names.size())); 

        printf("Inference DONE (memory attention).\n");
        */

  } else { //first frame (bypass memory attention)
     printf("No need to inference memory attention, the result is just the output of the image encoder \n");
     mem_attention_out.push_back(std::move(img_encoder_out[3]));     
  }

  //////////////////////
  // IMAGE DECODER
  /////////////////////
  //input_name= [point_coords] ===> [-1,-1,2]
  //input_name= [point_labels] ===> [-1,-1]
  //input_name= [image_embed] ===> [1,256,64,64]
  //input_name= [high_res_feats_0] ===> [1,32,256,256]
  //input_name= [high_res_feats_1] ===> [1,64,128,128]
  //output_name= [obj_ptr] ==> [-1,256]
  //output_name= [mask_for_mem] ==> [-1,1,-1,-1]
  //output_name= [pred_mask] ==> [-1,-1,-1,-1]

  //1.1) Create ORT Session (image decoder)
  printf("Creating ORT Session (image decoder)...\n");
  Ort::Env img_decoder_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING,"img_decoder");
  Ort::SessionOptions img_decoder_options = Ort::SessionOptions();
  img_decoder_options.SetIntraOpNumThreads(2);
  Ort::Session* img_decoder_session = new Ort::Session(img_decoder_env, "checkpoints/aimol/small/image_decoder.onnx", img_decoder_options);
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

  //COPY high_res_feat0 and high_res_feat1 from img_encoder_out to mem_attention_out
  printf("COPY high_res_feat0 and high_res_feat1 from img_encoder_out to mem_attention_out\n");
  mem_attention_out.push_back(std::move(img_encoder_out[1])); // high_res_feat0
  mem_attention_out.push_back(std::move(img_encoder_out[2])); // high_res_feat1
  //auto result_2 =this->img_decoder_infer(mem_attention_out);

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
  std::vector<Ort::Value> img_decoder_input_tensor;
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
  
  /*
  //AIXO PETA. NO ENTENC PER QUE HO POSA SI NO HI HA UN INPUT
  printf("Creating img_decoder_input_tensor (frame_size)\n");
  printf("\tNumber of elements in the data buffer = %lu\n", frame_size.size());
  printf("\tTensor shape dimensions = %lu\n", img_decoder_input_nodes[2].dim.size());
  img_decoder_input_tensor.push_back(Ort::Value::CreateTensor<int64>(
        memory_info, 
        frame_size.data(),
        frame_size.size(),
        img_decoder_input_nodes[2].dim.data(),
        img_decoder_input_nodes[2].dim.size()));
  */

  printf("Creating img_decoder_input_tensor (mem_attention_out)\n");
  img_decoder_input_tensor.push_back(std::move(mem_attention_out[0]));    // image_embed
  img_decoder_input_tensor.push_back(std::move(mem_attention_out[1]));    // high_res_feats_0
  img_decoder_input_tensor.push_back(std::move(mem_attention_out[2]));    // high_res_feats_1
  printf("Input tensors ready (image decoder).\n");

  

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
  std::vector<Ort::Value> output_tensors;
  //output_tensors.push_back(std::move(img_decoder_out[2])); //pred_mask
  output_tensors.push_back(std::move(img_decoder_out[tensorByName(img_decoder_out, img_decoder_output_names, "pred_mask")]));
  
  printf("Postprocessing...\n");
  postprocess(output_tensors, &image, box, point);
  //postprocess2(output_tensors, &image, box, point);
  cv::imshow("Image", image);
  cv::waitKey(0);
  printf("Postprocessing DONE.\n");
  
  /*
  //ryouchinsa
  const size_t maskInputSize = 256 * 256;
  std::vector<int64_t> orig_im_size_values_int64 = {ori_img_rows, ori_img_cols};
  cv::Mat outputMask = cv::Mat((int)orig_im_size_values_int64[0], (int)orig_im_size_values_int64[1], CV_8UC1, cv::Scalar(0));
  

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

  */
  printf("Postprocessing DONE.\n");
  
  return 0;
}
