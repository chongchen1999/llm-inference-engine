#include <iostream>
#include "src/utils/macro.h"
// #include "src/utils/debug_utils.h"
#include "src/layers/includes/context_decoder.h"

// Note: In LLaMA, all linear layers do not have bias.
// Note: I added `DeviceSyncAndCheckCudaError();` after many operations in the layers folder. 
// You can manually remove it or add conditional compilation code as shown in lesson30.

template <typename T>
void LlamaContextDecoder<T>::allocateMemoryForForward(LlamaAttentionDynamicParams *params) {
    const int num_tokens = params->num_tokens;
    const int batch_size = params->batch_size;
    const int max_q_len = params->max_q_len;
    const int max_k_len = params->max_k_len;

    const DataType type = getTensorType<T>();
    const DataType type_int = getTensorType<int>();

    decoder_residual = new TensorWrapper<T>(
        Device::GPU, type, {num_tokens, hidden_units}
    );

    attention_mask = new TensorWrapper<T>(
        Device::GPU, type, {batch_size, max_q_len, max_k_len}
    );

    padding_offset = new TensorWrapper<int>(
        Device::GPU, type_int, {batch_size, max_q_len}
    );

    cum_seqlens = new TensorWrapper<int>(
        Device::GPU, type_int, {batch_size + 1}
    );

    allocator->malloc(
        &decoder_residual->data,
        sizeof(T) * num_tokens * hidden_units, 
        false
    );

    allocator->malloc(
        &attention_mask->data,
        sizeof(T) * batch_size * max_q_len * max_k_len, 
        false
    );

    allocator->malloc(
        &padding_offset->data,
        sizeof(int) * batch_size * max_q_len, 
        false
    );

    allocator->malloc(
        &cum_seqlens->data,
        sizeof(int) * (batch_size + 1), 
        false
    );
}

template <typename T>
void LlamaContextDecoder<T>::freeBuf() {
    allocator->free(attention_mask->data);
    DeviceSyncAndCheckCudaError();

    allocator->free(padding_offset->data);
    DeviceSyncAndCheckCudaError();

    allocator->free(cum_seqlens->data);
    DeviceSyncAndCheckCudaError();
}

template <typename T>
void LlamaContextDecoder<T>::forward(
    TensorMap *input_tensors,
    const std::vector<LlamaLayerWeight<T> *> *layerWeights,
    TensorMap *output_tensors,
    LlamaAttentionDynamicParams *dyn_params
) {
    allocateMemoryForForward(dyn_params);

    Tensor *seq_lens = input_tensors->at("input_length");

    // 1. Calculate padding offset
    // Shape:
    // seq_lengths: [batch size]
    // output cum_seqlens: [batch size + 1], first element is 0
    // output padding_offset: [batch size * max q len]
    launchCalPaddingOffset(
        padding_offset,    // out
        cum_seqlens,       // out
        seq_lens->wrap<int>() // in
    );
    DeviceSyncAndCheckCudaError();

    // 2. Build causal mask
    Tensor *context_length = input_tensors->at("context_length");
    launchBuildCausalMasks<T>(
        attention_mask,            // out
        seq_lens->wrap<int>(),     // q, input lengths, [bs]
        context_length->wrap<int>() // k, context lengths, [bs]
    );
    DeviceSyncAndCheckCudaError();

    // 3. Context attention
    Tensor *history_length = input_tensors->at("history_length");
    Tensor *decoder_output = output_tensors->at("decoder_output");
    Tensor *all_k_cache = output_tensors->at("all_k_cache");
    Tensor *all_v_cache = output_tensors->at("all_v_cache");

    const DataType type_int = getTensorType<int>();
    const DataType type = getTensorType<T>();

    Tensor *layer_id = input_tensors->at("layer_id");
    Tensor *decoder_input = input_tensors->at("decoder_input");

    LLM_CHECK_WITH_INFO(
        decoder_input->wrap<T>()->data != nullptr, 
        "The data pointer of tensor inserted into TensorMap is nullptr!"
    );
    LLM_CHECK_WITH_INFO(
        history_length->wrap<int>()->data != nullptr, 
        "The data pointer of tensor inserted into TensorMap is nullptr!"
    );

    TensorMap context_attention_inputs{
        {"attention_input", decoder_input},
        {"padding_offset", padding_offset},
        {"history_length", history_length},
        {"input_length", seq_lens},
        {"context_length", context_length},
        {"attention_mask", attention_mask},
        {"layer_id", layer_id}
    };

    TensorMap context_attention_outputs{
        {"attention_output", decoder_output},
        {"all_k_cache", all_k_cache},
        {"all_v_cache", all_v_cache}
    };

    // Reuse the same buffer between layers
    for (int layer_id = 0; layer_id < num_layer; ++layer_id) {
        if (layer_id > 0) {
            TensorWrapper<int> *layer = new TensorWrapper<int>(
                Device::CPU, type_int, {1}, &layer_id
            );
            context_attention_inputs["layer_id"] = layer;
        }

        decoder_input = context_attention_inputs.at("attention_input");

        launchRMSNorm(
            decoder_input->wrap<T>(), // in & out, [num tokens, q_hidden_units]
            decoder_residual, // RMSNorm input hidden states, used for next add residual
            layerWeights->at(layer_id)->attn_norm_weight, // RMSNorm weights, [q_hidden_units]
            rmsnorm_eps
        );
        DeviceSyncAndCheckCudaError();

        context_attention->forward(
            context_attention_inputs,
            context_attention_outputs,
            layerWeights->at(layer_id)->self_attn_weight,
            dyn_params,
            context_attention->getAttnStaticParams()
        );

        launchFusedAddBiasResidualAndRMSNorm(
            decoder_residual, // [num tokens, hidden_units]
            decoder_output->wrap<T>(), // [num tokens, hidden_units]
            layerWeights->at(layer_id)->self_attn_weight.output, // bias
            layerWeights->at(layer_id)->ffn_norm_weight.gamma,   // RMSNorm weights, [hidden_units]
            rmsnorm_eps
        );
        DeviceSyncAndCheckCudaError();

        #ifdef SAVE_DATA
        save_tensor(decoder_output->as<T>(), "ffn_input.bin", layer_id);
        #endif

        TensorMap ffn_inputs{
            {"ffn_input", decoder_output}
        };
        TensorMap ffn_outputs{
            {"ffn_output", decoder_output}
        };

        // Used to distinguish FFN in context decoder or self decoder, to reduce print info
        dyn_params->is_ctx = true;

        ffn->forward(
            ffn_inputs, 
            ffn_outputs, 
            layerWeights->at(layer_id)->ffn_weight, 
            *dyn_params
        );

        #ifdef SAVE_DATA
        save_tensor(decoder_output->as<T>(), "ffn_output.bin", layer_id);
        #endif

        launchAddResidual(
            decoder_residual,            // residual, [num tokens, hidden_units]
            decoder_output->as<T>()      // in & out, [num tokens, hidden_units]
        );
        DeviceSyncAndCheckCudaError();

        context_attention_inputs["attention_input"] = decoder_output;
    }

    freeBuf();
    DeviceSyncAndCheckCudaError();
}

template class LlamaContextDecoder<float>;
template class LlamaContextDecoder<half>;
