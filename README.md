# sam2-c

IMPORTANT:
	- A back2black hi tinc sam2.1_tiny

LOG:
	(top more recent)
	- Adapting TensorCopy to XTENSOR. Need to use it yet...
	- Really installing XTENSOR will help!!
	- Implementing trunc_normal (considering using xtensor)
	- He aconseguit executar el MLP però m'ha calgut redimensionar el tensor
	- He aconseguit executar el memory encoder però cal:
		- verificar que el resultat és el mateix que en Python
		- he reescalat les màscares però hi ha dues opcions i no tinc clar si he triat la correcta (mirar codi)
	- DONE: Reutilitzar el output del prompt encoder
	- He provat els model d'ailia (tiny, 2.1, un frame) amb ryouchinsa i funciona :-)
	- El d'aimiol no sembla funcionar, estic mirant el de ryouchinsa que vaig fer servir a Lester (https://github.com/ryouchinsa/sam-cpp-macos)
	- Adaptant code d'aimiol pas a pas i provant-lo. Intentaré canviar-ho a ailia i provar.
	- Anotant aquí: SAM2.md
	- He creat OrtInferenceTEST i estic provant de reemplaçar les crides als models ONNX de ailia. 
	- A sam2-c estava creant C++ partint del que tenia a back2black però penso que:
		- Millor partir de OrtInference
		- Anar adaptant les parts una a una...
	- VERIFICAT que SAM2 amb ONNX de axinc-ai/ailia-models funciona amb video 
	- Descobert axinc-ai/ailia-models i aquests docs (BRUTAL):
		- https://deepwiki.com/axinc-ai/ailia-models/4.1-segment-anything-2
	- Alternativa: provar la versió onnx amb python de axinc-ai
	- Intentant WAY2 (Aimol-l), problemes per compilar

TODO:
	- WARNING amb la redimensió que faig del output del MLP
	- WARNING amb la redimensió que faig de les màscares d'entrada al memory encoder


1. WAY 1 (from scratch)

Test SAM2 Python locally

git clone https://github.com/facebookresearch/sam2.git

- first attempt to do it in python with onnx
	- The version of axinc-ai within the ailia-models repo is the best
	- I thing he did it: https://github.com/axinc-ai/segment-anything-2
	- https://github.com/axinc-ai/ailia-models/tree/master/image_segmentation/segment-anything-2
	- https://github.com/axinc-ai/ailia-models/issues/1514

	git clone https://github.com/axinc-ai/ailia-models.git
	cd ailia-models 
	python3.11 -m venv myvenv 
	source myvenv/bin/activate
	pip install -r requirements.txt
	pip install onnxruntime
	cd image_segmentation/segment-anything-2 
	python3 segment-anything-2.py --onnx    (downloads weights i tot)  
		(python3 segment-anything-2.py --onnx --model hiera_t --version 2.1)
	python3 segment-anything-2.py -v demo (works)
	python3 segment-anything-2.py -v demo --onnx (works)
	python3 segment-anything-2.py -v demo --onnx --model hiera_t (works)
	python3 segment-anything-2.py -v demo --onnx --model hiera_t --version 2.1 (works)

- test ailia-models-cpp  

	git clone https://github.com/axinc-ai/ailia-models-cpp.git
	cd ailia-models-cpp  

- add types (cython?)
- transpiler shed skin 



2. WAY 2 (Aimol-l)

NOTE: Only known working port with video (working with video reported here https://chatgpt.com/c/689304a0-c9e4-8325-8e7d-a1f82a38dc8b errors reported)

First clone the first repo (for obtaining the ONNX model): https://github.com/Aimol-l/SAM2Export

	git clone https://github.com/Aimol-l/SAM2Export

Then:

	cd SAM2Export
	cd checkpoints
	./download_ckpts.sh
	mkdir base+ large small tiny
	cp sam2_hiera_base_plus.pt sam2_hiera_base+.pt 

	cd ..
	python3.11 -m venv myvenv 
	source myvenv/bin/activate

	pip install "torch>=2.5.1"
	pip install "onnx==1.16.2" (l'usa a https://github.com/axinc-ai/segment-anything-2)
	pip install onnxsim
	pip install "hydra-core>=1.3.2"
	pip install "pillow>=9.4.0"
	pip install "tqdm>=4.66.1"

	python export_onnx.py

	deactivate
	cd $HOME/dev 

Then clone the second repo:

	git clone https://github.com/Aimol-l/OrtInference
	cd OrtInference

Copy the checkpoints/base+ folder from the first repo to the second repo

	cp -r ../SAM2Export/checkpoints/base+ models 

Instead of building like this:

	cmake -B build 
	cd build
	cmake .. && make && ../bin/./main

I created:

	./build_macos.sh

Problems with onnxruntime:
	- Changed include/algorithm/Model.h  
		onnxruntime/onnxruntime_cxx_api.h -> onnxruntime_cxx_api.h
	- Added onnxruntime folder from back2black
	- Some changes in CMakeLists.txt (see code)

Problems with result_of:
	- Commented:
	#if EIGEN_HAS_STD_RESULT_OF
	/*template<typename T> struct result_of {
	  typedef typename std::result_of<T>::type type1;
	  //typedef typename std::invoke_result_t<T>::type type1; //RUBEN
	  typedef typename remove_all<type1>::type type;
	};
	#else*/
	
	Ja que aquesta alternativa provocaba altres errors:
	Changed include/bytetrack/include/eigen/Eigen/src/Core/util/Meta.h:319:
		std::result_of -> std::invoke_result_t

Problem no troba  OSlibonnxruntime_providers_shared.dylib:

	(no l'hauria de buscar, era per culpa de compilar amb coses de CUDA)

	1) A main.cpp canviar:

	auto r = sam2->initialize(onnx_paths,true);
	->
	auto r = sam2->initialize(onnx_paths,false);

	2) Al main.cpp, al main activar només la línia sam2();

	3) 
	 	cd $HOME/dev/OrtInference
		mkdir -p models/sam2/small/
		cp models/base+/* models/sam2/small 
		mkdir -p assets/video
		(colocar un test.mkv allà) (he colocat footage.mp4 al final)

Problema executant: not enough space: expected 8388608, got 16
	hi ha un issue (https://github.com/Aimol-l/OrtInference/issues/23)

	Comentant aquesta línia del SAM2.cpp funciona el primer frame però després peta (no hi ha resposta al issue)

	input_tensor.push_back(Ort::Value::CreateTensor<int64>(memory_info,frame_size.data(),frame_size.size(),
                        this->img_decoder_input_nodes[2].dim.data(),
                        this->img_decoder_input_nodes[2].dim.size()));


Test with:

	cd bin (if exectuted from the root does not find the model file)
	./main 

3. WAY 3 (Nuitka)

4. WAY 4 (oter users)

kywish
liyihao76

@kywish (Author) on Nov 25, 2024 (edited)
I mainly referred to this: https://github.com/axinc-ai/segment-anything-2
I separated the image encoder, mask decoder, memory encoder, and memory attention. You need to manage ptr and feature caching yourself, but it’s not too complex.
The result is basically the same as the original, but CPU onnxruntime is too slow. I've been busy recently, so I haven’t had time to work on the GPU version 😂.
As for the prompt: if the user provides input in a frame, the label is the specified 0/1/2/3; if not, it is -1.

@weiyaokun on Dec 4, 2024
Partially resolved. Current progress:
 c++ onnxruntime full pipeline: CPU version, single target
 Multi-target
 fill_hole
 GPU version
How did you handle multi-target? Which part of the code needs to be modified?

@kywish (Author) on Dec 5, 2024
https://github.com/axinc-ai/segment-anything-2
This repo basically supports multi-target. Just change the rotation matrix encoding back to the 6-dim version.

@kyakuno
https://github.com/axinc-ai/ailia-models/issues/1514


PROBLEM WITH THE MEMORY ATTENTION MODULE:
https://github.com/facebookresearch/sam2/issues/186





3. LINKS

(Sep 11, 2024)

I implemented the export of all onnx files and realized the inference of video through c++ onnx runtime。
inference with C++ ：https://github.com/Aimol-l/OrtInference
export onnx file：https://github.com/Aimol-l/SAM2Export

https://github.com/Aimol-l/OrtInference ()
https://github.com/Aimol-l/SAM2Export ()

https://github.com/axinc-ai/segment-anything-2.git

https://github.com/ryouchinsa/sam-cpp-macos

https://github.com/DapengFeng/SAM-E/issues/1


5. TROUBLESHOOTING

- no member named 'make_unique' in namespace 'std'

	#include <onnxruntime_cxx_api.h>