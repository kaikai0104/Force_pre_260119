#include "neural_network.h"

/* 移动平均滤波 */
void moving_average(const float *input, float *output, int len, int window) {
    if (window <= 1) {
        memcpy(output, input, len * sizeof(float));
        return;
    }
    
    // 计算累积和
    float *cumsum = (float *)malloc((len + 1) * sizeof(float));
    cumsum[0] = 0.0f;
    for (int i = 0; i < len; i++) {
        cumsum[i + 1] = cumsum[i] + input[i];
    }
    
    // 计算移动平均
    int half_window = window / 2;
    for (int i = 0; i < len; i++) {
        if (i < half_window) {
            // 左边界: 使用第一个有效值
            int end = window < len ? window : len;
            output[i] = (cumsum[end] - cumsum[0]) / (float)end;
        } else if (i >= len - half_window) {
            // 右边界: 使用最后一个有效值
            int start = len - window > 0 ? len - window : 0;
            output[i] = (cumsum[len] - cumsum[start]) / (float)(len - start);
        } else {
            // 中间部分
            int start = i - half_window;
            int end = start + window;
            output[i] = (cumsum[end] - cumsum[start]) / (float)window;
        }
    }
    
    free(cumsum);
}

/* 计算数值梯度 */
void compute_gradient(const float *input, float *output, int len, float dt) {
    if (len <= 1) {
        output[0] = 0.0f;
        return;
    }
    
    // 前向差分 (第一个点)
    output[0] = (input[1] - input[0]) / dt;
    
    // 中心差分 (中间点)
    for (int i = 1; i < len - 1; i++) {
        output[i] = (input[i + 1] - input[i - 1]) / (2.0f * dt);
    }
    
    // 后向差分 (最后一个点)
    output[len - 1] = (input[len - 1] - input[len - 2]) / dt;
}

/* Play算子实现 */
void play_operator(const float *x_series, float *y_series, int len, float r) {
    if (len <= 0) return;
    
    y_series[0] = x_series[0];
    
    for (int t = 1; t < len; t++) {
        float delta = x_series[t] - y_series[t - 1];
        
        if (delta > r) {
            y_series[t] = x_series[t] - r;
        } else if (delta < -r) {
            y_series[t] = x_series[t] + r;
        } else {
            y_series[t] = y_series[t - 1];
        }
    }
}

/* 物理约束 */
void physical_force_constraint(float *force, int len, float dt, float max_rate, float min_force) {
    float max_change_per_step = max_rate * dt;
    
    // 应用变化率限制
    for (int i = 1; i < len; i++) {
        float delta = force[i] - force[i - 1];
        if (delta > max_change_per_step) {
            force[i] = force[i - 1] + max_change_per_step;
        } else if (delta < -max_change_per_step) {
            force[i] = force[i - 1] - max_change_per_step;
        }
    }
    
    // 应用非负约束
    for (int i = 0; i < len; i++) {
        if (force[i] < min_force) {
            force[i] = min_force;
        }
    }
}

/* 计算PI基线预测 */
void compute_pi_baseline(const float *x_smooth, const float *dx,
                         const float *iq_smooth, const float *diq,
                         float *output, int len) {
    // 分配临时数组用于存储Play算子输出
    float **P = (float **)malloc(NUM_PLAY_OPERATORS * sizeof(float *));
    for (int k = 0; k < NUM_PLAY_OPERATORS; k++) {
        P[k] = (float *)malloc(len * sizeof(float));
        play_operator(x_smooth, P[k], len, r_list[k]);
    }
    
    // 计算基线: F0 = sum(w_P * P) + beta_dx * dx + alpha_iq * iq_scaled + [beta_diq * diq_scaled] + bias
    for (int i = 0; i < len; i++) {
        float sum = 0.0f;
        
        // PI算子部分
        for (int k = 0; k < NUM_PLAY_OPERATORS; k++) {
            sum += w_P[k] * P[k][i];
        }
        
        // 位移导数
        sum += BETA_DX * dx[i];
        
        // 电流及其导数 (应用缩放)
        sum += ALPHA_IQ * (iq_smooth[i] * IQ_SCALE);
        
#if USE_DIQ
        sum += BETA_DIQ * (diq[i] * IQ_SCALE);
#endif
        
        // 偏置
        sum += PI_BIAS;
        
        output[i] = sum;
    }
    
    // 释放临时数组
    for (int k = 0; k < NUM_PLAY_OPERATORS; k++) {
        free(P[k]);
    }
    free(P);
}

/* 特征归一化 */
void normalize_features(const float *features, float *normalized) {
    for (int i = 0; i < INPUT_SIZE; i++) {
        float denom = f_max[i] - f_min[i];
        if (denom == 0.0f) {
            denom = 1.0f;
        }
        normalized[i] = 2.0f * (features[i] - f_min[i]) / denom - 1.0f;
    }
}

/* GRU前向传播 (单步) */
float gru_forward_step(const float *input, GRUState *state) {
    float r[HIDDEN_SIZE];  // reset gate
    float z[HIDDEN_SIZE];  // update gate
    float n[HIDDEN_SIZE];  // new gate
    
    // 计算三个门
    // r = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)
    // z = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)
    // n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))
    
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        // Reset gate
        float r_val = 0.0f;
        for (int i = 0; i < INPUT_SIZE; i++) {
            r_val += weight_ih_l0[h][i] * input[i];
        }
        r_val += bias_ih_l0[h];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            r_val += weight_hh_l0[h][j] * state->hidden[j];
        }
        r_val += bias_hh_l0[h];
        r[h] = sigmoid(r_val);
        
        // Update gate
        float z_val = 0.0f;
        for (int i = 0; i < INPUT_SIZE; i++) {
            z_val += weight_ih_l0[HIDDEN_SIZE + h][i] * input[i];
        }
        z_val += bias_ih_l0[HIDDEN_SIZE + h];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            z_val += weight_hh_l0[HIDDEN_SIZE + h][j] * state->hidden[j];
        }
        z_val += bias_hh_l0[HIDDEN_SIZE + h];
        z[h] = sigmoid(z_val);
        
        // New gate
        float n_val = 0.0f;
        for (int i = 0; i < INPUT_SIZE; i++) {
            n_val += weight_ih_l0[2 * HIDDEN_SIZE + h][i] * input[i];
        }
        n_val += bias_ih_l0[2 * HIDDEN_SIZE + h];
        
        float h_part = 0.0f;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            h_part += weight_hh_l0[2 * HIDDEN_SIZE + h][j] * state->hidden[j];
        }
        h_part += bias_hh_l0[2 * HIDDEN_SIZE + h];
        n_val += r[h] * h_part;
        n[h] = tanh_activation(n_val);
    }
    
    // 更新隐藏状态: h' = (1 - z) * n + z * h
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        state->hidden[h] = (1.0f - z[h]) * n[h] + z[h] * state->hidden[h];
    }
    
    // 全连接层输出
    float output = fc_bias;
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        output += fc_weight[h] * state->hidden[h];
    }
    
    return output;
}

/* GRU前向传播 (序列) */
float gru_forward_sequence(const float input_seq[][INPUT_SIZE], int seq_len) {
    GRUState state;
    memset(&state, 0, sizeof(GRUState));  // 初始化隐藏状态为0
    
    float output = 0.0f;
    for (int t = 0; t < seq_len; t++) {
        output = gru_forward_step(input_seq[t], &state);
    }
    
    return output;  // 返回最后一个时刻的输出
}

/* 完整的模型推理 */
void model_inference(const SampleData *samples, int num_samples,
                     float *predictions, int window_size) {
    // 分配临时数组
    float *x = (float *)malloc(num_samples * sizeof(float));
    float *iq = (float *)malloc(num_samples * sizeof(float));
    float *x_smooth = (float *)malloc(num_samples * sizeof(float));
    float *iq_smooth = (float *)malloc(num_samples * sizeof(float));
    float *dx = (float *)malloc(num_samples * sizeof(float));
    float *diq = (float *)malloc(num_samples * sizeof(float));
    float *dx_smooth = (float *)malloc(num_samples * sizeof(float));
    float *diq_smooth = (float *)malloc(num_samples * sizeof(float));
    float *F0 = (float *)malloc(num_samples * sizeof(float));
    float *mode_flag = (float *)malloc(num_samples * sizeof(float));
    
    // 提取原始数据
    for (int i = 0; i < num_samples; i++) {
        x[i] = samples[i].x;
        iq[i] = samples[i].iq;
    }
    
    // 移动平均平滑
    moving_average(x, x_smooth, num_samples, window_size);
    moving_average(iq, iq_smooth, num_samples, window_size);
    
    // 计算梯度
    compute_gradient(x_smooth, dx, num_samples, DT);
    compute_gradient(iq_smooth, diq, num_samples, DT);
    
    // 对梯度再次平滑
    moving_average(dx, dx_smooth, num_samples, window_size);
    moving_average(diq, diq_smooth, num_samples, window_size);
    
    // 计算PI基线
    compute_pi_baseline(x_smooth, dx_smooth, iq_smooth, diq_smooth, F0, num_samples);
    
    // 应用物理约束到基线预测
#if APPLY_PHYSICAL_CONSTRAINT
    physical_force_constraint(F0, num_samples, DT, FORCE_MAX_RATE, FORCE_MIN);
#endif
    
    // 计算模式标志: dx < -thr 且 |iq| > ratio*max|iq|
    float iq_abs_max = 0.0f;
    for (int i = 0; i < num_samples; i++) {
        float abs_iq = fabsf(iq_smooth[i] * IQ_SCALE);
        if (abs_iq > iq_abs_max) {
            iq_abs_max = abs_iq;
        }
    }
    iq_abs_max += 1e-12f;
    
    float mode_dx_thr = 1e-5f;
    float mode_iq_ratio = 0.02f;
    for (int i = 0; i < num_samples; i++) {
        int cond1 = (dx_smooth[i] < -mode_dx_thr);
        int cond2 = (fabsf(iq_smooth[i] * IQ_SCALE) > mode_iq_ratio * iq_abs_max);
        mode_flag[i] = (float)(cond1 && cond2);
    }
    
    // GRU残差预测
    float input_seq[SEQ_LEN][INPUT_SIZE];
    float residual_norm;
    
    for (int i = 0; i < num_samples; i++) {
        if (i < SEQ_LEN - 1) {
            // 序列不足,使用基线预测
            predictions[i] = F0[i];
        } else {
            // 构建输入序列: [x, dx, iq, diq, F0, mode]
            for (int t = 0; t < SEQ_LEN; t++) {
                int idx = i - SEQ_LEN + 1 + t;
                float features[INPUT_SIZE] = {
                    x_smooth[idx],
                    dx_smooth[idx],
                    iq_smooth[idx] * IQ_SCALE,
                    diq_smooth[idx] * IQ_SCALE,
                    F0[idx],
                    mode_flag[idx]
                };
                normalize_features(features, input_seq[t]);
            }
            
            // GRU前向传播
            residual_norm = gru_forward_sequence(input_seq, SEQ_LEN);
            
            // 反标准化残差
            float residual = residual_norm * RESIDUAL_STD + RESIDUAL_MEAN;
            
            // 最终预测 = 基线 + 残差
            predictions[i] = F0[i] + residual;
        }
    }
    
    // 应用物理约束到最终预测
#if APPLY_PHYSICAL_CONSTRAINT
    physical_force_constraint(predictions, num_samples, DT, FORCE_MAX_RATE, FORCE_MIN);
#endif
    
    // 释放临时数组
    free(x);
    free(iq);
    free(x_smooth);
    free(iq_smooth);
    free(dx);
    free(diq);
    free(dx_smooth);
    free(diq_smooth);
    free(F0);
    free(mode_flag);
}

/* 计算评估指标 */
void compute_metrics(const float *predictions, const float *ground_truth, int len,
                     float *mae, float *rmse, float *r2) {
    // MAE和RMSE
    float sum_abs_error = 0.0f;
    float sum_squared_error = 0.0f;
    
    for (int i = 0; i < len; i++) {
        float error = predictions[i] - ground_truth[i];
        sum_abs_error += fabsf(error);
        sum_squared_error += error * error;
    }
    
    *mae = sum_abs_error / (float)len;
    *rmse = sqrtf(sum_squared_error / (float)len);
    
    // R²
    float mean_gt = 0.0f;
    for (int i = 0; i < len; i++) {
        mean_gt += ground_truth[i];
    }
    mean_gt /= (float)len;
    
    float ss_tot = 0.0f;
    for (int i = 0; i < len; i++) {
        float diff = ground_truth[i] - mean_gt;
        ss_tot += diff * diff;
    }
    
    *r2 = 1.0f - (sum_squared_error / (ss_tot + 1e-12f));
}
