#include "src/models/llama/llama.h"

// ()note: we only support batch size = 1 now
// ！！()note:
// 目前暂时只支持输入"Hey, are you conscious? Can you talk to me?"，支持了dynamic input shape后方可支持其它输入
// C++ tokenizer Encode暂时不能正常运行，正在fix，故以上输入暂时只能手动通过HF tokenizer API获取，见src/tools/HF_llama_run_script.py
// cpu unpinned buffer
template <typename T>
void LlamaModel<T>::allocateCPUBuffer(int batch_size) {
    allocator->malloc(&h_input_ids_buf_, sizeof(int) * 13, true);
    allocator->malloc(&h_input_length_buf_, sizeof(int) * batch_size, true);
    allocator->malloc(&h_history_length_buf_, sizeof(int) * batch_size, true);
    allocator->malloc(&h_context_length_buf_, sizeof(int) * batch_size, true);
    allocator->malloc(&h_sequence_lengths_, sizeof(int) * batch_size, true);
    allocator->malloc(&h_finished_buf_, sizeof(bool) * batch_size, true);
    for (int i = 0; i < batch_size; ++i) {
        h_finished_buf_[i] = 0;
    }
    h_output_ids_ = allocator->malloc(h_output_ids_, sizeof(int) * batch_size, true);
}

// alloc gpu buffer
template <typename T>
void LlamaModel<T>::allocateGPUBuffer(int batch_size) {
    // Initialize tensor wrappers
    step = new TensorWrapper<int>(Device::CPU, getTensorType<int>(), {1});
    layer = new TensorWrapper<int>(Device::CPU, getTensorType<int>(), {1}, &layer_id);

    // For context decoder
    context_decoder_input = new TensorWrapper<T>(Device::GPU, getTensorType<T>(), {13, hidden_units});
    context_decoder_output = new TensorWrapper<T>(Device::GPU, getTensorType<T>(), {13, hidden_units});
    context_decoder_lmhead_input = new TensorWrapper<T>(Device::GPU, getTensorType<T>(), {1, hidden_units});

    // For self decoder
    decoder_input = new TensorWrapper<T>(Device::GPU, getTensorType<T>(), {1, hidden_units});
    decoder_output = new TensorWrapper<T>(Device::GPU, getTensorType<T>(), {1, hidden_units});

    input_ids = new TensorWrapper<int>(Device::GPU, getTensorType<int>(), {13}); 

    // Context decoder lengths
    input_length = new TensorWrapper<int>(Device::GPU, getTensorType<int>(), {batch_size});
    history_length = new TensorWrapper<int>(Device::GPU, getTensorType<int>(), {batch_size});
    context_length = new TensorWrapper<int>(Device::GPU, getTensorType<int>(), {batch_size});
    sequence_lengths = new TensorWrapper<int>(Device::GPU, getTensorType<int>(), {batch_size});

    // KV cache buffer
    all_k_cache = new TensorWrapper<T>(Device::GPU, getTensorType<T>(), {num_layers, batch_size, kv_head_num, max_seq_len, head_size});
    all_v_cache = new TensorWrapper<T>(Device::GPU, getTensorType<T>(), {num_layers, batch_size, kv_head_num, max_seq_len, head_size});

    token_ids = new TensorWrapper<int>(Device::GPU, getTensorType<int>(), {batch_size});
    is_finished = new TensorWrapper<bool>(Device::GPU, getTensorType<bool>(), {batch_size});
    output_rmsnorm_weight = new TensorWrapper<T>(Device::GPU, getTensorType<T>(), {hidden_units}, llama_weights->out_rmsnorm_weight.gamma);
    probs = new TensorWrapper<T>(Device::GPU, getTensorType<T>(), {batch_size, vocab_size});
    unused_residual = new TensorWrapper<T>(Device::GPU, getTensorType<T>(), {batch_size, hidden_units});

    // Allocate buffers
    allocator->malloc(&unused_residual->data, sizeof(T) * 13 * hidden_units);
    allocator->malloc(&context_decoder_input->data, sizeof(T) * 13 * hidden_units);
    allocator->malloc(&context_decoder_output->data, sizeof(T) * 13 * hidden_units);
    allocator->malloc(&context_decoder_lmhead_input->data, sizeof(T) * 1 * hidden_units);
    allocator->malloc(&decoder_input->data, sizeof(T) * batch_size * hidden_units);
    allocator->malloc(&decoder_output->data, sizeof(T) * batch_size * hidden_units);
    allocator->malloc(&input_ids->data, sizeof(int) * 13);
    allocator->malloc(&input_length->data, sizeof(int) * batch_size);
    allocator->malloc(&history_length->data, sizeof(int) * batch_size);
    allocator->malloc(&context_length->data, sizeof(int) * batch_size);
    allocator->malloc(&sequence_lengths->data, sizeof(int) * batch_size);
    allocator->malloc(&all_k_cache->data, sizeof(T) * num_layers * batch_size * max_seq_len * kv_head_num * head_size);
    allocator->malloc(&all_v_cache->data, sizeof(T) * num_layers * batch_size * max_seq_len * kv_head_num * head_size);
    allocator->malloc(&token_ids->data, sizeof(int) * batch_size);
    allocator->malloc(&is_finished->data, sizeof(bool) * batch_size);
    allocator->malloc(&probs->data, sizeof(T) * batch_size * vocab_size);

    // TopK buffers
    topk_id = new TensorWrapper<int>(Device::GPU, getTensorType<int>(), {batch_size, beamwidth, blocks_per_beam, K});
    topk_val = new TensorWrapper<T>(Device::GPU, getTensorType<T>(), {batch_size, beamwidth, blocks_per_beam, K});
    final_topk_id = new TensorWrapper<int>(Device::GPU, getTensorType<int>(), {batch_size * beamwidth, K});
    final_topk_val = new TensorWrapper<T>(Device::GPU, getTensorType<T>(), {batch_size * beamwidth, K});

    allocator->malloc(&topk_id->data, sizeof(int) * batch_size * beamwidth * blocks_per_beam * K);
    allocator->malloc(&topk_val->data, sizeof(T) * batch_size * beamwidth * blocks_per_beam * K);
    allocator->malloc(&final_topk_id->data, sizeof(int) * batch_size * beamwidth * K);
    allocator->malloc(&final_topk_val->data, sizeof(T) * batch_size * beamwidth * K);
}

// free CPU and GPU buffer
template <typename T>
void LlamaModel<T>::free() {
    allocator->free(h_input_ids_buf_, true);
    allocator->free(h_input_length_buf_, true);
    allocator->free(h_history_length_buf_, true);
    allocator->free(h_context_length_buf_, true);
    allocator->free(h_sequence_lengths_, true);
    DeviceSyncAndCheckCudaError();

    allocator->free(context_decoder_input->data);
    allocator->free(context_decoder_output->data);
    allocator->free(decoder_input->data);
    allocator->free(decoder_output->data);
    allocator->free(input_ids->data);
    DeviceSyncAndCheckCudaError();
    
    allocator->free(input_length->data);
    allocator->free(history_length->data);
    allocator->free(context_length->data);
    allocator->free(sequence_lengths->data);
    allocator->free(all_k_cache->data);
    allocator->free(all_v_cache->data);
    allocator->free(token_ids->data);
    allocator->free(is_finished->data);
    allocator->free(probs->data);
    DeviceSyncAndCheckCudaError();
}

template <typename T>
void LlamaModel<T>::initializeForContextDecoder(MapStringToInt &int_params_first_token) {
    // only support and assumed bs = 1
    h_input_length_buf_[0] = int_params_first_token["cur_input_length"];
    h_history_length_buf_[0] = int_params_first_token["history_length"];
    h_context_length_buf_[0] = int_params_first_token["context_length"];

    // get from tokenizer encode
    CHECK(cudaMemcpy(input_ids->data, h_input_ids_buf_, sizeof(int) * h_input_length_buf_[0], cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(input_length->data, h_input_length_buf_, sizeof(int) * batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(history_length->data, h_history_length_buf_, sizeof(int) * batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(context_length->data, h_context_length_buf_, sizeof(int) * batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(is_finished->data, h_finished_buf_, sizeof(bool) * batch_size, cudaMemcpyHostToDevice));
}

template <typename T>
void LlamaModel<T>::initializationForSelfDecoder() {
    // nothing to do now
}

// () note: 返回所有轮次总共的input、总共input中的history部分、总共input中的当前轮次input部分
template <typename T>
std::vector<std::string> LlamaModel<T>::makeInput(
    const std::string &history, 
    int round, 
    const std::string &input
) {
    std::vector<std::string> ret = {(round == 0 ? "" : history) + input, history, input};
    return ret;
}

template <typename T>
// () note: 根据第round轮的结果制作history
std::string LlamaModel<T>::makeHistory(
    const std::string &history, 
    int round, 
    const std::string &input, 
    const std::string &output
) {
    return (round == 0 ? prompt : history) + input + output; 
}

// () note: input embedding kernel wrapper
template <typename T>
void LlamaModel<T>::inputEmbedding(TensorWrapper<int> *input_ids, TensorWrapper<T> *decoder_input) {
    launchInputEmbedding<T>(input_ids, decoder_input, &(llama_weights->pre_decoder_embedding_weight));
    DeviceSyncAndCheckCudaError();
}

// () note: 每轮对话的1st token generation for context decoder
template <typename T>
int LlamaModel<T>::generateFirstToken(
    LlamaAttentionDynamicParams &dparams, 
    MapStringToInt &int_params_first_token
) {
    initializeForContextDecoder(int_params_first_token);
    inputEmbedding(input_ids, context_decoder_input);
    LLM_CHECK_WITH_INFO(context_decoder_input->data != nullptr, "GPU context decoder input data is not initialized");
    LLM_CHECK_WITH_INFO(history_length->data != nullptr, "GPU history_length data is not initialized");
    LLM_CHECK_WITH_INFO(input_length->data != nullptr, "GPU input_length data is not initialized");
    LLM_CHECK_WITH_INFO(context_length->data != nullptr, "GPU context_length data is not initialized");
    LLM_CHECK_WITH_INFO(output_rmsnorm_weight->data != nullptr, "GPU output_rmsnorm_weight data is not initialized");

    TensorMap decoder_inputs{
        {"decoder_input", context_decoder_input},
        {"history_length", history_length},
        {"input_length", input_length},
        {"context_length", context_length},
        {"output_norm_weight", output_rmsnorm_weight}, // located at llamaweights class, rather not llamalayerweigths
        {"layer_id", layer}
    };

    // output buffer and input buffer are shared to reuse buffer between layers
    // I dont rewrite Tensor's copy constructor, default shallow copy, that can share buffer, which is I want
    TensorMap decoder_outputs{
        {"decoder_output", context_decoder_output},
        {"all_k_cache", all_k_cache},
        {"all_v_cache", all_v_cache}
    };

    context_decoder->forward(
        decoder_inputs,
        llama_weights->llama_layer_weight, // layerWeights,
        decoder_outputs,
        dparams
    );

    // output rmsnorm
    Tensor* decoder_output = decoder_outputs["decoder_output"];
    launchRMSNorm(
        decoder_output->wrap<T>(), //in&out, [bs, q_hidden_units]
        unused_residual,
        llama_weights->out_rmsnorm_weight,//rmsnorm weights, [q_hidden_units]
        rmsnorm_eps,
        true
    );

    saveTensor(decoder_output->wrap<T>() ,"decoder_norm_out.bin");
    DeviceSyncAndCheckCudaError();
    int res = LMHeadAndTopKSample(decoder_outputs);
    //std::cout << "context decoder generated  index  is " << res << "\n";
    return res;
}

template <typename T>
int LlamaModel<T>::generateNextToken(LlamaAttentionDynamicParams &dparams) {
    initializationForSelfDecoder();
    inputEmbedding(input_ids, decoder_input);
    TensorMap decoder_inputs{
        {"decoder_input", decoder_input},
        {"step", step}, // a batch shared same step, locate on CPU, no need GPU
        {"finished", is_finished},
        {"layer_id", layer},
        {"output_norm_weight", output_rmsnorm_weight} // located at llamaweights class, rather not llamalayerweigths
    };
    // () note: 最开始是context decoder里面RoPE输出的k和v写到kv cache
    // () note: self decoder之后每一个step都会输出kv到kv cache, 需要保证kv cache是llama class的成员, 这样就可以保证同步更新
    TensorMap decoder_outputs{
        {"decoder_output", decoder_output},
        {"all_k_cache", all_k_cache},
        {"all_v_cache", all_v_cache}
    };

    self_decoder->forward(
        decoder_inputs,
        llama_weights->llama_layer_weight,
        decoder_outputs,
        dparams
    );

    // output rmsnorm
    Tensor *decoder_output = decoder_outputs["decoder_output"];
    launchRMSNorm(
        decoder_output->wrap<T>(), //in&out, [bs, q_hidden_units]
        unused_residual,
        llama_weights->out_rmsnorm_weight,//rmsnorm weights, [q_hidden_units]
        rmsnorm_eps,
		true
    );
    DeviceSyncAndCheckCudaError();
    int res = LMHeadAndTopKSample(decoder_outputs);
    return res;
}

template <typename T>
int LlamaModel<T>::LMHeadAndTopKSample(TensorMap &decoder_outputs) {
    Tensor *decoder_output = decoder_outputs["decoder_output"];
    if (index == 0) {
        TensorWrapper<T> *decoder_output_tensorwrapper = decoder_output->wrap<T>();
        const auto input_length = decoder_output_tensorwrapper->shape[0];
        const auto hidden_units = decoder_output_tensorwrapper->shape[1];

        // Fetch last token to handle ctxdecoder sampling
        const T *ptr = decoder_output_tensorwrapper->data + (input_length - 1) * hidden_units;
        context_decoder_lmhead_input->data = ptr;

        launchLinearGemm(
            context_decoder_lmhead_input,                      // [1, hidden_units] for ctx decoder
            llama_weights->post_decoder_embedding_weight,      // lm_head.weight.bin, [vocab_size, hidden_units]
            probs,                                             // [1, vocab_size] for context decoder
            cublas_wrapper,
            false,
            true
        );
        DeviceSyncAndCheckCudaError();
    } else {
        // For self decoder
        launchLinearGemm(
            decoder_output->wrap<T>(),                           // [bs, hidden_units] for self decoder
            llama_weights->post_decoder_embedding_weight,      // lm_head.weight.bin, [vocab_size, hidden_units]
            probs,                                             // [bs, vocab_size] for self decoder
            cublas_wrapper,
            false,
            true
        );
        DeviceSyncAndCheckCudaError();
    }

    launchTopKForBeamSearch(
        probs,                // [bs, vocab_size]
        topk_id,
        topk_val,
        final_topk_id,
        final_topk_val       // Output, this is a temporary buffer defined in allocatebuffer
    );
    DeviceSyncAndCheckCudaError();

    int_params_of_sample.insert({"step", *step->data});

    launchSampling(
        final_topk_id,       // In
        final_topk_val,      // In
        sequence_lengths,    // Out, +1
        is_finished,         // Out, to determine if finished
        token_ids,           // Out, newly generated token ids
        int_params_of_sample // In, including step, vocab size, end id
    );
    DeviceSyncAndCheckCudaError();

    CHECK(cudaMemcpy(h_output_ids_, token_ids->data, sizeof(int) * batch_size, cudaMemcpyDeviceToHost));

    // Return the generated index (only for bs = 1)
    return h_output_ids_[0];
}

// 单轮对话, batch size = 1
// 返回所有轮次总共的input、总共input中的history部分、总共input中的当前轮次input部分
template <typename T>
std::string LlamaModel<T>::response(
    const std::vector<std::string> &input,
    CallBack PrintRes
) {
    // Temporary hardcoded token IDs for testing
    std::vector<int> token_ids = {1, 18637, 29892, 526, 366, 19861, 29973, 1815, 366, 5193, 304, 592, 29973};
    
    std::string history_str = input[1];
    std::vector<int> history_input_ids;
    if (!history_str.empty()) {
        history_input_ids = tokenizer.Encode(history_str);
    }

    std::string total_str = input[0];
    std::vector<int> context_ids;
    if (!total_str.empty()) {
        // context_ids = tokenizer.Encode(total_str);
        context_ids = {1, 18637, 29892, 526, 366, 19861, 29973, 1815, 366, 5193, 304, 592, 29973};
    }

    std::copy(token_ids.begin(), token_ids.end(), h_input_ids_buf_.begin());

    // Ensure all needed input buffers are prepared
    int ret;
    int context_length = static_cast<int>(context_ids.size());
    int history_length = static_cast<int>(history_input_ids.size());
    int cur_input_length = static_cast<int>(token_ids.size()); // Real input length

    MapStringToInt int_params_first_token;
    int_params_first_token["context_length"] = context_length;
    int_params_first_token["history_length"] = history_length;
    int_params_first_token["cur_input_length"] = cur_input_length;

    LlamaAttentionDynamicParams attn_dyn_params;
    attn_dyn_params.batch_size = 1;
    attn_dyn_params.num_tokens = cur_input_length;
    attn_dyn_params.max_q_len = cur_input_length; // Max length of q in one batch
    attn_dyn_params.max_k_len = context_length;   // Max context length

    step->data = &context_length; // Used in self-decoder phase

    std::string ret_string;
    for (int index = 0; index < output_token_limit; ++index) { // Self-defined output token limit
        if (index == 0) {
            ret = generateFirstToken(attn_dyn_params, int_params_first_token);
        } else {
            ret = generateNextToken(attn_dyn_params);
            if (ret == eos_token_id) {
                break;
            }
        }

        ++(*step->data);
        std::string gen_string = tokenizer.Decode({ret});
        ret_string += gen_string;
        PrintRes(index, gen_string.c_str());

        if (index == 0) {
            TensorWrapper<int> tmp(CPU, getTensorType<int>(), {1}, &ret);
            LLM_CHECK(tmp.shape != input_ids->shape);
            LLM_CHECK(tmp.dtype == input_ids->dtype);
            LLM_CHECK(tmp.location != input_ids->location);

            allocator->Free(input_ids->data);
            input_ids->data = allocator->Malloc(input_ids->data, sizeof(int) * 1, false);
            input_ids->shape = {1};

            CHECK(cudaMemcpy(input_ids->data, tmp.data, sizeof(int) * 1, cudaMemcpyHostToDevice));
        } else {
            CHECK(cudaMemcpy(input_ids->data, &ret, sizeof(int) * 1, cudaMemcpyHostToDevice));
        }
    }

    PrintRes(-1, ret_string.c_str());
    return ret_string;
}

template class LlamaModel<float>;
template class LlamaModel<half>;