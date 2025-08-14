///////////
// AIMOL //
///////////

OrtInference:

image_encoder
memory_attention
image_decoder
memory_encoder

Code in SAM2.cpp:

image_encoder->mem_attention->img_decoder->mem_encoder
	

///////////
// AILIA //
///////////

	//DOC: https://deepwiki.com/axinc-ai/ailia-models/4.1-segment-anything-2
	propagate_in_video_preflight
		_consolidate_temp_output_across_obj
			high_res_masks = interpolate(consolidated_out["pred_masks"],(self.image_size, self.image_size))
			_run_memory_encoder
				_get_image_feature
				_encode_new_memory (ONNX call within)
					inputs: pix_feat and mask_for_mem
				(optional) _get_maskmem_pos_enc
		results:
			consolidated_out["maskmem_features"] = maskmem_features
            consolidated_out["maskmem_pos_enc"] = maskmem_pos_enc

(grandària 1024x1024)
def interpolate(low_res_multimasks, image_size):
    high_res_multimasks = np.zeros((low_res_multimasks.shape[0], low_res_multimasks.shape[1], image_size[0], image_size[1]), dtype=np.float32)
    for b in range(low_res_multimasks.shape[0]):
        for c in range(low_res_multimasks.shape[1]):
            high_res_multimasks[b][c] = cv2.resize(low_res_multimasks[b][c], (image_size[1], image_size[0]), high_res_multimasks, interpolation=cv2.INTER_LINEAR)
    return high_res_multimasks

