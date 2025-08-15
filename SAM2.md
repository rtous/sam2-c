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
	
	frame_idx = 0
    while (True):
		image = preprocess_frame ---- ENCODER ----
		if frame_idx == 0:
			annotate_frame
				propagate_in_video_preflight
					for frame_idx in temp_frame_inds:
						consolidated_out = self._consolidate_temp_output_across_obj
							_consolidate_temp_output_across_obj
								high_res_masks = interpolate(consolidated_out["pred_masks"],(self.image_size, self.image_size))
								_run_memory_encoder (optionally??) ---- MEM ENCODER ----
									_get_image_feature
									_encode_new_memory (ONNX call within)
										inputs: pix_feat and mask_for_mem
												(optional) _get_maskmem_pos_enc
										outputs:
											consolidated_out["maskmem_features"] = maskmem_features
								            consolidated_out["maskmem_pos_enc"] = maskmem_pos_enc
	                			output_dict[storage_key][frame_idx] = consolidated_out
	                			_add_output_per_object()

	    process_frame
	    	propagate_in_video
	    		_run_single_frame_inference 
					track_step (also from _get_empty_mask_ptr)
						_prepare_memory_conditioned_features 
							if first frame:
								 pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
								 pix_feat_with_mem = np.transpose(pix_feat_with_mem, (1, 2, 0)).reshape(B, C, H, W)
					             # Use a dummy token on the first frame (to avoid empty memory input to tranformer encoder)
					            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
					            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]
					        else:
					        	THE TRICKY PART
					        	obj_ptr_tpos_proj.run --- OBJ_PTR_TPOS_PROJ ---

							memory = np.concatenate(to_cat_memory, axis=0)
					        memory_pos_embed = np.concatenate(to_cat_memory_pos_embed, axis=0)

							memory_1 = memory[:-num_obj_ptr_tokens,:,:]
					        memory_2 = memory[-num_obj_ptr_tokens:,:,:]
					        memory_pos_embed_1 = memory_pos_embed[:-num_obj_ptr_tokens,:,:]
					        memory_pos_embed_2 = memory_pos_embed[-num_obj_ptr_tokens:,:,:]
					        attention_mask_1 = np.zeros((memory_1.shape[0], memory_1.shape[1]), dtype=np.bool_)
					        attention_mask_2 = np.zeros((memory_2.shape[0], memory_2.shape[1]), dtype=np.bool_)
					        attention_mask_1[:memory_1.shape[0],:] = True
					        attention_mask_2[:memory_2.shape[0],:] = True
							pix_feat_with_mem = memory_attention.run ---- MEM ATTENTION ----
							# reshape the output (HW)BC => BCHW
					        pix_feat_with_mem = np.transpose(pix_feat_with_mem, (1, 2, 0)).reshape(B, C, H, W)
				        _forward_sam_heads() ()

				        	prompt_encoder.run ---- PROMPT ENCODER ----

							mask_decoder.run()  ---- DECODER ----

							forward_postprocess

							mlp.run ---- MLP ---- # Extract object pointer from the output token
	
			        

	
(grandària 1024x1024)
def interpolate(low_res_multimasks, image_size):
    high_res_multimasks = np.zeros((low_res_multimasks.shape[0], low_res_multimasks.shape[1], image_size[0], image_size[1]), dtype=np.float32)
    for b in range(low_res_multimasks.shape[0]):
        for c in range(low_res_multimasks.shape[1]):
            high_res_multimasks[b][c] = cv2.resize(low_res_multimasks[b][c], (image_size[1], image_size[0]), high_res_multimasks, interpolation=cv2.INTER_LINEAR)
    return high_res_multimasks

