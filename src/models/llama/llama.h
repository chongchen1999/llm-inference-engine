#include "../basemodel.h"
#include "llama_params.h"
#include "../../weights/includes/llama_weights.h"
#include "../../layers/includes/context_attention.h"
#include "../../layers/includes/self_decoder.h"
#include "../../kernels/includes/input_embedding.h"  // embedding
#include "../../kernels/includes/linear.h"           // LM Head
#include "../../kernels/includes/topk.h"             // topK
#include "../../kernels/includes/sampling.h"         // sampling
#include "../../models/tokenizer.h"
#include "../../utils/debug_utils.h"

template<typename T>
class LlamaModel : public BaseModel {
private:
    int head_num;
    int kv_head_num;
    int head_size;
    int inter_size;
    int num_layers;
    int vocab_size;
    int vocab_size_padded;
    float rmsnorm_eps = 1e-5f;
    int hidden_units;
    int max_seq_len;  // self-defined
    int output_token_limit = 20;  // self-defined
    int pad_token_id = 0;  // from hf modeling_config
    int bos_token_id = 1;
    int eos_token_id = 2;
    int layer_id = 0;
    int batch_size = 1;  // can be included in dynamic parameters or not
    int beamwidth = 1;   // needed by beam search; set 1 by default when not using beam search
    int blocks_per_beam = 8;  // needed by topK
    int index = 0;
    std::string prompt = "";  // self-defined or not

    Tokenizer tokenizer;
    std::unique_ptr<LlamaWeight<T>> llama_weights;
    std::unique_ptr<LlamaContextDecoder<T>> context_decoder;
    std::unique_ptr<LlamaSelfDecoder<T>> self_decoder;

    int K = 4;  // K of topK sort
    std::unique_ptr<TensorWrapper<int>> step;
    std::unique_ptr<TensorWrapper<T>> output_rmsnorm_weight;
    std::unique_ptr<TensorWrapper<int>> layer;
    std::unique_ptr<TensorWrapper<T>> context_decoder_input;
    std::unique_ptr<TensorWrapper<T>> context_decoder_output;
    std::unique_ptr<TensorWrapper<T>> context_decoder_lmhead_input;
    std::unique_ptr<TensorWrapper<T>> decoder_input;
    std::unique_ptr<TensorWrapper<T>> decoder_output;

    std::unique_ptr<TensorWrapper<int>> input_ids;
    std::unique_ptr<TensorWrapper<int>> input_length;
    std::unique_ptr<TensorWrapper<int>> history_length;
    std::unique_ptr<TensorWrapper<int>> context_length;

    std::unique_ptr<TensorWrapper<T>> all_k_cache;
    std::unique_ptr<TensorWrapper<T>> all_v_cache;
    std::unique_ptr<TensorWrapper<T>> unused_residual;
    
    // Used by sampling
    MapStringToInt int_params_of_sample;
    std::unique_ptr<TensorWrapper<T>> probs;
    std::unique_ptr<TensorWrapper<int>> token_ids;
    std::unique_ptr<TensorWrapper<int>> sequence_lengths;  // Record current sequence length in GENERATE
    std::unique_ptr<TensorWrapper<bool>> is_finished;
    std::unique_ptr<TensorWrapper<int>> topk_id;
    std::unique_ptr<TensorWrapper<T>> topk_val;
    std::unique_ptr<TensorWrapper<int>> final_topk_id;
    std::unique_ptr<TensorWrapper<T>> final_topk_val;

    // Pinned or not pinned CPU buffers
    int *h_input_ids_buf_ = nullptr;
    int *h_input_length_buf_ = nullptr;
    int *h_history_length_buf_ = nullptr;
    int *h_context_length_buf_ = nullptr;
    int *h_sequence_lengths_ = nullptr;
    bool *h_finished_buf_ = nullptr;
    int *h_output_ids_ = nullptr;

public:
    LlamaModel() = default;

    LlamaModel(
        int head_num,
        int kv_head_num,
        int head_size,
        int inter_size,
        int num_layers,
        int vocab_size,
        const LlamaAttentionStaticParams &attention_static_params,
        int max_seq_len,
        cudaStream_t stream,
        CublasWrapper *cublas_wrapper,
        BaseAllocator *allocator,
        CudaDeviceProp *cuda_device_prop
    ) :
        BaseModel(stream, cublas_wrapper, allocator, cuda_device_prop),
        head_num(head_num),
        kv_head_num(kv_head_num),
        head_size(head_size),
        inter_size(inter_size),
        num_layers(num_layers),
        vocab_size(vocab_size),
        vocab_size_padded(vocab_size),
        hidden_units(head_num * head_size),
        max_seq_len(max_seq_len) {
        
        int_params_of_sample.insert({"vocab_size", vocab_size});
        int_params_of_sample.insert({"end_id", eos_token_id});
        layer = std::make_unique<TensorWrapper<int>>(Device::CPU, DataType::INT32, {1}, &layer_id);
        
        llama_weights = std::make_unique<LlamaWeight<T>>(
            head_num,
            kv_head_num,
            head_size,
            inter_size,
            vocab_size,
            num_layers,
            false,  // attn_bias
            getWeightType<T>()
        );

        self_decoder = std::make_unique<LlamaSelfDecoder<T>>(
            head_num,
            kv_head_num,
            head_size,
            inter_size,
            num_layers,
            attention_static_params,
            rmsnorm_eps,
            stream,
            cublas_wrapper,
            allocator
        );

        context_decoder = std::make_unique<LlamaContextDecoder<T>>(
            head_num,
            kv_head_num,
            head_size,
            inter_size,
            num_layers,
            attention_static_params,
            rmsnorm_eps,
            stream,
            cublas_wrapper,
            allocator
        );

        // Allocate buffers
        allocateCPUBuffer(1);  // bs = 1
        allocateGPUBuffer(1);
    }

    ~LlamaModel() {
        free();
    }

    void loadTokenizer(const std::string &file) {
        tokenizer.Initialize(file);
    }

    void loadWeights(const std::string &file) {
        llama_weights->loadWeights(file);
    }

    void loadWeightsFromDummy() {
        llama_weights->loadWeightsFromDummy();
    }

    void allocateCPUBuffer(int batch_size);
    void allocateGPUBuffer(int batch_size);
    void free();

    std::vector<std::string> makeInput(
        const std::string &history,
        int round,
        const std::string &input
    );
    
    std::string makeHistory(
        const std::string &history,
        int round,
        const std::string &input,
        const std::string &output
    );
    
    std::string response(
        const std::vector<std::string> &input,
        CallBack PrintRes
    );

    int makeOutput();

    void inputEmbedding(
        std::unique_ptr<TensorWrapper<int>> &input_ids,
        std::unique_ptr<TensorWrapper<T>> &decoder_input
    );
    
    void initializeForContextDecoder(
        MapStringToInt &int_params_first_token
    );

    int generateFirstToken(
        LlamaAttentionDynamicParams &dparams,
        MapStringToInt &int_params_first_token
    );

    void initializationForSelfDecoder();

    int generateNextToken(LlamaAttentionDynamicParams &dparams);

    int LMHeadAndTopKSample(TensorMap &decoder_outputs);
};
