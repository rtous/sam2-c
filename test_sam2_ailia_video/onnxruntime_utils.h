#ifndef ONNXRUNTIME_UTILS_H_
#define ONNXRUNTIME_UTILS_H_

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <list>
#include <fstream>
#include <iostream>
#include <cstring>
#include "xtensor/containers/xarray.hpp"
#include "xtensor/containers/xadapt.hpp"


//Information about a input/output"node" or "port of the ONNX model.
//Parts of the shape can be unknown (-1), dynamic.
//Similar information can be obtained from a tensor (see printTensorInfo).
struct InputOutput {
        int input_or_ouput;//0 input, 1 output
        char* name;
        ONNXTensorElementDataType type;
        std::vector<int64_t> shape; // may contain -1 for dynamic dims
};

//shape vector to string
static std::string shape2string(std::vector<int64_t> shape) {
    std::string dim_str = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        dim_str += std::to_string(shape[i]);
        if (i != shape.size() - 1) dim_str += ",";
    }
    dim_str += "]";
    return dim_str;
}

/*
    Ort::Value is move-only and represents ownership of the underlying tensor memory.
    When you assign (=) or insert into vector (push_back) you’re transfer ownership, the source becomes empty;
    So, if you want to reuse the same Ort::Value need to copy the contents in a TensorCopy.
    I use it, e.g., for the ouput of the prompt encoder (obtained just once from the first frame but used for every frame).
*/
class TensorCopy {
public:
  InputOutput info;
  std::vector<float> data;
  int size;
};

class TensorCopyXTENSOR {
public:
  InputOutput info;
  xt::xarray<float> data;
  int size;
};


TensorCopyXTENSOR setTensorCopyXTENSOR(Ort::Value src) {
  TensorCopyXTENSOR tcopy;
  auto tensor_info = src.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> shape = tensor_info.GetShape();
  auto src_size = tensor_info.GetElementCount();
  auto type = tensor_info.GetElementType();
  if (type != 1) {
    printf("ERROR: Trying to use setTensorCopy with a non-float tensor.\n");
    exit(-1);
  }
  tcopy.info.shape = shape;
  tcopy.info.type = type;
  tcopy.size = src_size;
  //std::vector<float> obj_ptrs(src_size); // first+recent // 16*256
  //tcopy.data.resize(src_size);
  const float* tensor_data = src.GetTensorData<float>();
  //std::copy_n(tensor_data, src_size, std::begin(tcopy.data));

  // Wrap ONNX Runtime data as a view
  //const float* tensor_data = src.GetTensorData<float>();
  std::vector<size_t> shape_size_t(shape.begin(), shape.end());
  auto view = xt::adapt(tensor_data, src_size, xt::no_ownership(), shape_size_t);

  tcopy.data = xt::xarray<float>(view);//deep copy owns its memory
  return tcopy;
}

TensorCopy setTensorCopy(Ort::Value src) {
  TensorCopy tcopy;
  auto tensor_info = src.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> shape = tensor_info.GetShape();
  auto src_size = tensor_info.GetElementCount();
  auto type = tensor_info.GetElementType();
  if (type != 1) {
    printf("ERROR: Trying to use setTensorCopy with a non-float tensor.\n");
    exit(-1);
  }
  tcopy.info.shape = shape;
  tcopy.info.type = type;
  tcopy.size = src_size;
  //std::vector<float> obj_ptrs(src_size); // first+recent // 16*256
  tcopy.data.resize(src_size);
  const float* tensor_data = src.GetTensorData<float>();
  std::copy_n(tensor_data, src_size, std::begin(tcopy.data));
  return tcopy;
}

//Don't need to change it at all.
Ort::Value getTensorCopy(TensorCopy& tcopy, Ort::MemoryInfo& memory_info) {
  auto tensor = Ort::Value::CreateTensor<float>(
                  memory_info,
                  tcopy.data.data(),
                  tcopy.data.size(),
                  tcopy.info.shape.data(),
                  tcopy.info.shape.size()
                  );
  return tensor;
}

static const char* OnnxTypeToString(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return "uint8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return "int8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  return "uint16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:   return "int16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:  return "string";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    return "bool";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  return "float64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  return "uint32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:  return "uint64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:  return "complex64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128: return "complex128";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:   return "bfloat16";
        default: return "unknown";
    }
}

//TO BE DELETED, BETTER USE printTensorCopyInfo
void printTensorInfo(Ort::Value src) {
  auto tensor_info = src.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> shape = tensor_info.GetShape();
  auto src_size = tensor_info.GetElementCount();
  auto type = tensor_info.GetElementType();
  if (type != 1) {
    printf("ERROR: Trying to use setTensorCopy with a non-float tensor.\n");
    exit(-1);
  }
  printf("num elements = [%d]\n", src_size);
  printf("shape = [%s]\n", shape2string(shape).c_str());
  printf("type = [%s]\n", OnnxTypeToString(type));
  std::vector<float> data(src_size);
  const float* tensor_data = src.GetTensorData<float>();
  std::copy_n(tensor_data, src_size, std::begin(data));
  /*for(int i=0; i<shape.size(); i++) {
    for(int j=0; j<shape[i]; j++) {
        for(int k=0; k<) {
            printf("%f, ", data[]);
        }
    }
  }*/
  printf("10 elements = ");
  for(int i=0; i<10; i++) {
    printf("%f, ", data[i]);
  }

}

/*
TensorCopy slice_1xNxC_to1x1xC(TensorCopy tcopy) {
  TensorCopy tcopy_out;
  int C = tcopy.info.shape[tcopy.info.shape.size()-1];
  tcopy_out.info.shape = tcopy.info.shape;
  tcopy_out.info.shape[1] = 1;
  tcopy_out.info.type = tcopy.info.type;
  tcopy_out.size = C;
  tcopy_out.data.resize(tcopy_out.size);
  std::copy_n(tcopy.data.data(), tcopy_out.size, std::begin(tcopy_out.data));
  return tcopy;
}*/

void printTensorCopyInfo(TensorCopy tcopy) {
  printf("num elements = [%d]\n", tcopy.size);
  printf("shape = [%s]\n", shape2string(tcopy.info.shape).c_str());
  printf("type = [%s]\n", OnnxTypeToString(tcopy.info.type));
  printf("10 elements = ");
  /*for(int i=0; i<10; i++) {
    printf("%f, ", tcopy.data[i]);
  }*/

}

//NOT USED YET
/*
TensorCopy slice_1xNxC_to1x1xC(TensorCopy tcopy) {
  TensorCopy tcopy_out;
  int C = tcopy.info.shape[tcopy.info.shape.size()-1];
  tcopy_out.info.shape = tcopy.info.shape;
  tcopy_out.info.shape[1] = 1;
  tcopy_out.info.type = tcopy.info.type;
  tcopy_out.size = C;
  tcopy_out.data.resize(tcopy_out.size);
  std::copy_n(tcopy.data.data(), C, std::begin(tcopy_out.data));
  return tcopy_out;
}
*/

//Used to reshape the outut of MLP
TensorCopy slice_1xNxC_toNxC(TensorCopy tcopy) {
  TensorCopy tcopy_out;
  tcopy_out.info.shape = tcopy.info.shape;
  tcopy_out.info.shape.erase(tcopy_out.info.shape.begin()); 
  tcopy_out.info.type = tcopy.info.type;
  tcopy_out.size = tcopy.size;
  tcopy_out.data.resize(tcopy_out.size);
  std::copy_n(tcopy.data.data(), tcopy_out.size, std::begin(tcopy_out.data));
  return tcopy_out;
}

//Not used but also works to reshape the outut of MLP so keep if previous fail
TensorCopy slice_1xNxC_to1xC(TensorCopy tcopy) {
  TensorCopy tcopy_out;
  tcopy_out.info.shape = tcopy.info.shape;
  int C = tcopy.info.shape[tcopy.info.shape.size()-1];
  tcopy_out.info.shape.erase(tcopy_out.info.shape.begin()); 
  tcopy_out.info.type = tcopy.info.type;
  tcopy_out.size = tcopy.size;
  tcopy_out.data.resize(tcopy_out.size);
  std::copy_n(tcopy.data.data(), C, std::begin(tcopy_out.data));
  return tcopy_out;
}

class OrtModel {
public:
    OrtModel(const std::string& model_name, const std::string& model_path, const std::string& device = "CPU", int threadsNumber = 2)
        : 
        env_(ORT_LOGGING_LEVEL_WARNING, model_name.c_str()), 
        session_options_(), 
        allocator_(), 
        session_(nullptr) {

        printf("/******** model %s:\n", model_name.c_str());
        session_options_.SetIntraOpNumThreads(2);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        if (device == "cpu") {
            //Do nothing
        } else if (device == "cuda") {
            int gpuDeviceId = std::stoi(device.substr(5));
            OrtCUDAProviderOptions options;
            options.device_id = gpuDeviceId;
            session_options_.AppendExecutionProvider_CUDA(options);
        }
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);

        load_io_info();
    }
    

    // Run with inputs (Ort::Value). Returns vector<Ort::Value> outputs
    std::vector<Ort::Value> run(const std::vector<Ort::Value>& inputs) {
        Ort::RunOptions run_options;

        std::vector<const char*> input_names = getInputNames();
        std::vector<const char*> output_names = getOutputNames();

        if (inputs.size() != input_names.size()) {
            throw std::runtime_error("Number of inputs does not match model definition");
        }

        auto outputs = std::move(session_->Run(
          Ort::RunOptions{ nullptr },
          input_names.data(),    //input names
          inputs.data(),   //input tensor
          inputs.size(),   //input tensor size
          output_names.data(),   //output names
          output_names.size())); //output tensors size

        return outputs;
    }

    std::vector<const char*> getInputNames() {
        std::vector<const char*> input_names;
        for (int i = 0; i < inputs.size(); ++i) {
            InputOutput info = inputs[i];
            input_names.push_back(info.name);
        }
        return input_names;
    }

    std::vector<const char*> getOutputNames() {
        std::vector<const char*> output_names;
        for (int i = 0; i < outputs.size(); ++i) {
            InputOutput info = outputs[i];
            output_names.push_back(info.name);
        }
        return output_names;
    }

    int inputIdxByName(char* name_queried) {
        std::vector<const char*> input_names = getInputNames();
        int i;
        for (i=0; i<input_names.size() && strcmp(input_names[i], name_queried) != 0; i++) {
            //printf("Checked %s != %s\n", input_names[i], name_queried);
        }
        if (i<input_names.size()) {
            return i;
        } else {
            printf("ERROR, input tensor name %s not found. \n", name_queried);
            exit(-1);
        }  
    }

    int outputIdxByName(char* name_queried) {
        std::vector<const char*> output_names = getOutputNames();
        int i;
        for (i=0; i<output_names.size() && strcmp(output_names[i], name_queried) != 0; i++) {
            //printf("Checked %s != %s\n", output_names[i], name_queried);
        }
        if (i<output_names.size()) {
            return i;
        } else {
            printf("ERROR, output tensor name %s not found. \n", name_queried);
            exit(-1);
        }   
    }
    
    void load_io_info() {
        size_t num_input_nodes = session_->GetInputCount();
        size_t num_output_nodes = session_->GetOutputCount();

        for (size_t i = 0; i < num_input_nodes; ++i) {
            Ort::AllocatedStringPtr output_name_Ptr = session_->GetInputNameAllocated(i, allocator_);
            const char* name = output_name_Ptr.get();
            Ort::TypeInfo type_info = session_->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            InputOutput info;
            info.input_or_ouput = 0;
            int name_length = strlen(name) + 1;
            info.name = new char[name_length];
            strcpy(info.name , name);
            info.type = tensor_info.GetElementType();
            info.shape = tensor_info.GetShape();
            inputs.push_back(info);
            printf("input_name= [%s] ===> %s ===> %s\n", info.name, shape2string(info.shape).c_str(), OnnxTypeToString(info.type));
        }

        for (size_t i = 0; i < num_output_nodes; ++i) {
            Ort::AllocatedStringPtr output_name_Ptr = session_->GetOutputNameAllocated(i, allocator_);
            const char* name = output_name_Ptr.get();
            Ort::TypeInfo type_info = session_->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            InputOutput info;
            info.input_or_ouput = 1;
            int name_length = strlen(name) + 1;
            info.name = new char[name_length];
            strcpy(info.name , name);
            info.type = tensor_info.GetElementType();
            info.shape = tensor_info.GetShape();
            outputs.push_back(info);
            printf("output_name= [%s] ===> %s ===> %s\n", info.name, shape2string(info.shape).c_str(), OnnxTypeToString(info.type));
        }
    }

    Ort::Session* session() { return session_.get(); }

    std::vector<InputOutput> inputs; 
    std::vector<InputOutput> outputs;


private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::unique_ptr<Ort::Session> session_;
};

#endif
