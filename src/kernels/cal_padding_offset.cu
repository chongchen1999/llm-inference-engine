#include "src/kernels/includes/cal_padding_offset.h"

// Shape:
// - seq_lengths: [batch size]
// - cum_seqlens: [batch size + 1], first element is 0
// - padding_offset: [batch size * max_q_len]
// Note: The point is to calculate padding offset and cumulative offset

/*
    For example:
    input_lengths: [4, 3, 5]
    cum_seqlens: [0, 4, 7, 12]
    padding_offset: [0, 0, 0, 0,
                     1, 1, 1,
                     3, 3, 3, 3, 3]
*/
__global__ void calPaddingOffset(
    int *padding_offset,    // [batch_size, max_q_len]
    int *cum_seqlens,      // [batch_size + 1]
    const int *input_lengths, // [batch_size]
    const int batch_size,
    const int max_q_len
) {
    int tot_seqlen = 0;
    int tot_padding = 0;
    int idx = 0;

    for (int i = 0; i < batch_size; ++i) {
        int cur_seqlen = input_lengths[i];
        int cur_padding = max_q_len - cur_seqlen;

        cum_seqlens[i] = tot_seqlen;

        for (int j = 0; j < cur_seqlen; ++j) {
            padding_offset[idx++] = tot_padding;
        }

        tot_padding += cur_padding;
        tot_seqlen += cur_seqlen;
    }

    cum_seqlens[batch_size] = tot_seqlen;
}

void launchCalPaddingOffset(
    TensorWrapper<int> *padding_offset,    // [batch_size, max_q_len]
    TensorWrapper<int> *cum_seqlens,      // [batch_size + 1]
    TensorWrapper<int> *input_lengths     // [batch_size]
) {
    const int batch_size = padding_offset->shape[0];
    const int max_q_len = padding_offset->shape[1];

    LLM_CHECK_WITH_INFO(
        batch_size == input_lengths->shape[0],
        "Input lengths numbers should equal to padding offset batch size dimension!"
    );

    LLM_CHECK_WITH_INFO(
        batch_size == cum_seqlens->shape[0] - 1,
        "Cumulative sequence length should equal to padding offset batch size dimension plus 1!"
    );

    calPaddingOffset<<<1, 1>>>(
        padding_offset->data,
        cum_seqlens->data,
        input_lengths->data,
        batch_size,
        max_q_len
    );
}