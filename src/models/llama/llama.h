#include "../basemodel.h"
#include "llama_params.h"
#include "../../weights/llama/llama_weights.h"
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
    const int head_num;
    const int kv_head_num;
    const int head_size;
    const int inter_size;
    const int num_layers;
    const int vocab_size;
    int vocab_size_padded;
    const float rmsnorm_eps = 1e-5f;
    const int hidden_units; 
    const int max_seq_len;  // self-defined
    const int output_token_limit = 20;  // self-defined
    const int pad_token_id = 0;  // from hf modeling_config 
    const int bos_token_id = 1;
    const int eos_token_id = 2;
    const int layer_id = 0;
    const int batch_size = 1;  // can be included in dynamic parameters or not
    const int beamwidth = 1;   // needed by beam search; set 1 by default when not using beam search
    const int blocks_per_beam = 8;  // needed by topK
    const int index = 0;
    std::string prompt = "";  // self-defined or not

    Tokenizer tokenizer;
    LlamaWeight<T> *llama_weights;
    LlamaContextDecoder<T> *context_decoder;
    LlamaSelfDecoder<T> *self_decoder;

    int K = 4;  // K of topK sort
    TensorWrapper<int> *step;
    TensorWrapper<T> *output_rmsnorm_weight;
    TensorWrapper<int> *layer;
    TensorWrapper<T> *context_decoder_input;
    TensorWrapper<T> *context_decoder_output;
    TensorWrapper<T> *context_decoder_lmhead_input;
    TensorWrapper<T> *decoder_input;
    TensorWrapper<T> *decoder_output;

    TensorWrapper<int> *input_ids;
    TensorWrapper<int> *input_length;
    TensorWrapper<int> *history_length;
    TensorWrapper<int> *context_length;

    TensorWrapper<T> *all_k_cache;
    TensorWrapper<T> *all_v_cache;
    TensorWrapper<T> *unused_residual;
    // Used by sampling
    MapStringToInt int_params_of_sample;
    TensorWrapper<T> *probs;
    TensorWrapper<int> *token_ids;
    TensorWrapper<int> *sequence_lengths;  // Record current sequence length in GENERATE
    TensorWrapper<bool> *is_finished;
    TensorWrapper<int> *topk_id;
    TensorWrapper<T> *topk_val;
    TensorWrapper<int> *final_topk_id;
    TensorWrapper<T> *final_topk_val;

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
        const LlamaAttentionStaticParams &attn_static_params,
        int max_seq_len,
        cudaStream_t stream,
        cublasWrapper *cublas_wrapper,
        BaseAllocator *allocator,
        cudaDeviceProp *cuda_device_prop
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
        layer = new TensorWrapper<int>(Device::CPU, DataType::INT32, {1}, &layer_id);
        llama_weights = new LlamaWeight<T>(
            head_num,
            kv_head_num,
            head_size,
            inter_size,
            vocab_size,
            num_layers,
            false,  // attn_bias
            getWeightType<T>()
        );

        self_decoder = new LlamaSelfDecoder<T>(
            head_num,
            kv_head_num,
            head_size,
            inter_size,
            num_layers,
            attn_static_params,
            rmsnorm_eps,
            stream,
            cublas_wrapper,
            allocator
        );

        context_decoder = new LlamaContextDecoder<T>(
            head_num,
            kv_head_num,
            head_size,
            inter_size,
            num_layers,
            attn_static_params,
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
        this->free();
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

    std::vector<std::string> makeInput(const std::string &history, int round, const std::string &input);
    std::string makeHistory(const std::string &history, int round, const std::string &input, const std::string &output);
    std::string response(const std::vector<std::string> &input, CallBack PrintRes);

    int makeOutput();

    void inputEmbedding(TensorWrapper<int> *input_ids, TensorWrapper<T> *decoder_input);
    void initializeForContextDecoder(MapStringToInt &int_params_first_token);
    int generateFirstToken(
        LlamaAttentionDynamicParams &dparams, 
        MapStringToInt &int_params_first_token
    );
    void initializationForSelfDecoder();
    int generateNextToken(LlamaAttentionDynamicParams &dparams);
    int LMHeadAndTopKSample(TensorMap &decoder_outputs);
};
