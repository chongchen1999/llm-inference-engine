#pragma once

struct LlamaAttentionStaticParams {
    int rotary_embedding_dim;
    float rotary_embedding_base;
    int max_position_embeddings;
    bool use_dynamic_ntk; // for dyn scaling rope
    int head_size = 128;
    int head_num = 32;
    int kv_head_num = 32;
};

// note: llama类模型里面动态改变的变量, 注意非全部必需
struct LlamaAttentionDynamicParams {
    int batch_size;
    int num_tokens;
    int max_q_len;
    int max_k_len;
    int num_layers;
    bool is_context = false;
};