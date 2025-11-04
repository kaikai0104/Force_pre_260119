/*
 * 主程序: 验证C语言实现的神经网络模型
 * 读取Excel数据，调用模型推理，输出验证结果
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "neural_network.h"

// 简单的CSV读取函数 (Excel需要转换为CSV)
// 或者使用libxlsxwriter库读取Excel
int read_csv_data(const char *filename, SampleData **samples, int *num_samples) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        printf("Error: Cannot open file %s\n", filename);
        return -1;
    }
    
    // 先统计行数
    char line[1024];
    int count = 0;
    
    // 跳过表头
    if (fgets(line, sizeof(line), fp) == NULL) {
        fclose(fp);
        return -1;
    }
    
    while (fgets(line, sizeof(line), fp)) {
        count++;
    }
    
    if (count == 0) {
        printf("Error: No data in file\n");
        fclose(fp);
        return -1;
    }
    
    // 分配内存
    *samples = (SampleData *)malloc(count * sizeof(SampleData));
    if (*samples == NULL) {
        printf("Error: Memory allocation failed\n");
        fclose(fp);
        return -1;
    }
    
    // 重新读取数据
    rewind(fp);
    fgets(line, sizeof(line), fp);  // 跳过表头
    
    int i = 0;
    while (fgets(line, sizeof(line), fp) && i < count) {
        float x, iq, F;
        if (sscanf(line, "%f,%f,%f", &x, &iq, &F) == 3) {
            (*samples)[i].x = x;
            (*samples)[i].iq = iq;
            (*samples)[i].F = F;
            i++;
        }
    }
    
    fclose(fp);
    *num_samples = i;
    
    printf("Successfully loaded %d samples from %s\n", i, filename);
    return 0;
}

// 保存结果到CSV文件
int save_results_csv(const char *filename, const SampleData *samples, 
                     const float *predictions, int num_samples) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        printf("Error: Cannot create file %s\n", filename);
        return -1;
    }
    
    // 写表头
    fprintf(fp, "point,Time,x,iq,true_value,predicted_value,error\n");
    
    // 写数据
    for (int i = 0; i < num_samples; i++) {
        float time = i * 0.002f;  // 2ms采样间隔
        float error = predictions[i] - samples[i].F;
        fprintf(fp, "%d,%.6f,%.10f,%.10f,%.6f,%.6f,%.6f\n",
                i, time, samples[i].x, samples[i].iq, 
                samples[i].F, predictions[i], error);
    }
    
    fclose(fp);
    printf("Results saved to %s\n", filename);
    return 0;
}

int main(int argc, char *argv[]) {
    printf("==============================================\n");
    printf("  Neural Network Model Validation (C Implementation)\n");
    printf("  Model: PI Operator + GRU Residual Network\n");
    printf("==============================================\n\n");
    
    // 检查命令行参数
    if (argc < 2) {
        printf("Usage: %s <input_csv_file> [output_csv_file]\n", argv[0]);
        printf("\nNote: Input file should be CSV format with columns: x, iq, F\n");
        printf("      You can convert Excel to CSV using Python or Excel itself\n");
        return 1;
    }
    
    const char *input_file = argv[1];
    const char *output_file = (argc >= 3) ? argv[2] : "validation_results.csv";
    
    // 打印模型参数信息
    printf("Model Configuration:\n");
    printf("  Input Size: %d\n", INPUT_SIZE);
    printf("  Hidden Size: %d\n", HIDDEN_SIZE);
    printf("  Sequence Length: %d\n", SEQ_LEN);
    printf("  Number of PI Operators: %d\n", NUM_PLAY_OPERATORS);
    printf("  Sampling Period: %.4f s\n", DT);
    printf("  Physical Constraint: %s\n", APPLY_PHYSICAL_CONSTRAINT ? "Enabled" : "Disabled");
    if (APPLY_PHYSICAL_CONSTRAINT) {
        printf("    Max Force Rate: %.1f N/s\n", FORCE_MAX_RATE);
        printf("    Min Force: %.1f N\n", FORCE_MIN);
    }
    printf("\n");
    
    // 读取数据
    SampleData *samples = NULL;
    int num_samples = 0;
    
    printf("Loading data from %s...\n", input_file);
    if (read_csv_data(input_file, &samples, &num_samples) != 0) {
        return 1;
    }
    
    // 分配预测结果数组
    float *predictions = (float *)malloc(num_samples * sizeof(float));
    if (predictions == NULL) {
        printf("Error: Memory allocation failed\n");
        free(samples);
        return 1;
    }
    
    // 执行模型推理
    printf("\nRunning model inference...\n");
    int window_size = 5;  // 移动平均窗口大小
    model_inference(samples, num_samples, predictions, window_size);
    printf("Inference completed.\n\n");
    
    // 提取真实值
    float *ground_truth = (float *)malloc(num_samples * sizeof(float));
    for (int i = 0; i < num_samples; i++) {
        ground_truth[i] = samples[i].F;
    }
    
    // 计算评估指标
    float mae, rmse, r2;
    compute_metrics(predictions, ground_truth, num_samples, &mae, &rmse, &r2);
    
    printf("Validation Results:\n");
    printf("  Number of samples: %d\n", num_samples);
    printf("  MAE (Mean Absolute Error): %.3f N\n", mae);
    printf("  RMSE (Root Mean Square Error): %.3f N\n", rmse);
    printf("  R² Score: %.4f\n", r2);
    printf("\n");
    
    // 计算并显示统计信息
    float min_pred = predictions[0], max_pred = predictions[0];
    float min_true = ground_truth[0], max_true = ground_truth[0];
    for (int i = 1; i < num_samples; i++) {
        if (predictions[i] < min_pred) min_pred = predictions[i];
        if (predictions[i] > max_pred) max_pred = predictions[i];
        if (ground_truth[i] < min_true) min_true = ground_truth[i];
        if (ground_truth[i] > max_true) max_true = ground_truth[i];
    }
    
    printf("Force Statistics:\n");
    printf("  Ground Truth Range: [%.2f, %.2f] N\n", min_true, max_true);
    printf("  Prediction Range: [%.2f, %.2f] N\n", min_pred, max_pred);
    printf("\n");
    
    // 保存结果
    printf("Saving results to %s...\n", output_file);
    save_results_csv(output_file, samples, predictions, num_samples);
    
    // 释放内存
    free(samples);
    free(predictions);
    free(ground_truth);
    
    printf("\nValidation completed successfully!\n");
    printf("==============================================\n");
    
    return 0;
}
