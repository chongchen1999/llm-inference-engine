#include <iostream>
#include "../utils/macro.h"
#include "includes/self_decoder.h"
#include <vector>
#include "../memory/memory_deleter.cuh"

template<typename T>
void LlamaSelfDecoder<T>::allocateMemory(LlamaAttentionDynamicParams *dynamic_params) {
    const DataType type = getTensorType<T>();
    const int batch_size = dynamic_params->batch_size;

    decoder_residual = new TensorWrapper<T>(Device::GPU, type, std::vector<int>{batch_size, hidden_units});
    allocator->malloc(&decoder_residual->data, sizeof(T) * batch_size * hidden_units, false);
}

template<typename T>
void LlamaSelfDecoder<T>::freeBuf() {
    allocator->free(decoder_residual->data);
    DeviceSyncAndCheckCudaError();

    deallocate(decoder_residual, "new");
}

template<typename T>
void LlamaSelfDecoder<T>::forward(
    TensorMap *input_tensors,
    std::vector<LlamaLayerWeight<T> *> *layerWeights,
    TensorMap *output_tensors,
    LlamaAttentionDynamicParams *dynamic_params
) {
    allocateMemory(dynamic_params);

    Tensor *decoder_input = input_tensors->at("decoder_input");
    Tensor *step = input_tensors->at("step");
    Tensor *finished = input_tensors->at("finished");
    Tensor *decoder_output = output_tensors->at("decoder_output");
    Tensor *all_k_cache = output_tensors->at("all_k_cache");
    Tensor *all_v_cache = output_tensors->at("all_v_cache");
    Tensor *layer_id = input_tensors->at("layer_id");

    DataType type_int = getTensorType<int>();

    LLM_CHECK_WITH_INFO(
        decoder_input->wrap<T>()->data != nullptr,
        "The data pointer of tensor inserted into TensorMap is nullptr!"
    );
    LLM_CHECK_WITH_INFO(
        step->wrap<int>()->data != nullptr,
        "The data pointer of tensor inserted into TensorMap is nullptr!"
    );
    LLM_CHECK_WITH_INFO(
        finished->wrap<bool>()->data != nullptr,
        "The data pointer of tensor inserted into TensorMap is nullptr!"
    );

    TensorMap self_attention_inputs{
        {"attention_input", decoder_input},
        {"layer_id", layer_id},
        {"step", step},
        {"finished", finished}
    };

    TensorMap self_attention_outputs{
        {"attention_output", decoder_output},
        {"all_k_cache", all_k_cache},
        {"all_v_cache", all_v_cache}
    };

    for (int layer_id = 0; layer_id < num_layer; ++layer_id) {
        if (layer_id > 0) {
            auto layer = new TensorWrapper<int>(Device::CPU, type_int, std::vector<int>{1}, &layer_id);
            self_attention_inputs.insert({"layer_id", layer});
        }

        decoder_input = self_attention_inputs.at("attention_input");

        launchRMSNorm(
            decoder_input->wrap<T>(),  // in&out, [bs, q_hidden_units]
            decoder_residual,  // rmsnorm input hidden states, as input of next add residual
            &layerWeights->at(layer_id)->attention_norm_weight,  // rmsnorm weights, [q_hidden_units]
            rmsnorm_eps
        );
        DeviceSyncAndCheckCudaError();

        self_attention->forward(
            &self_attention_inputs,
            &self_attention_outputs,
            &layerWeights->at(layer_id)->self_attention_weight,
            dynamic_params
        );

        launchFusedAddBiasResidualAndRMSNorm(
            decoder_residual,  // in residual from tensor before rmsnorm and return decoder_residual + decoder_output, [bs, q_hidden_units]
            decoder_output->wrap<T>(),  // in&out from attention output, [bs, q_hidden_units]
            &layerWeights->at(layer_id)->self_attention_weight.output,  // bias
            layerWeights->at(layer_id)->ffn_norm_weight.gamma,  // rmsnorm weights, [q_hidden_units]
            rmsnorm_eps
        );
        DeviceSyncAndCheckCudaError();

        TensorMap ffn_inputs{{"ffn_input", decoder_output}};
        TensorMap ffn_outputs{{"ffn_output", decoder_output}};

        ffn->forward(
            &ffn_inputs,
            &ffn_outputs,
            &layerWeights->at(layer_id)->ffn_weight,
            dynamic_params
        );

        launchAddResidual(
            decoder_residual,  // in, [bs, hidden_units]
            decoder_output->wrap<T>(),  // in&out, [bs, hidden_units]
            true
        );
        DeviceSyncAndCheckCudaError();

        self_attention_inputs.insert({"attention_input", decoder_output});  // for next iteration
    }

    // No intermediate buffer to free, so ignoring free call
}

template class LlamaSelfDecoder<float>;
template class LlamaSelfDecoder<half>;
