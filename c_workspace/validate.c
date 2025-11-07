/*
 * PI-GRU 神经网络模型 - 验证工具
 * 模型: PI算子基线 + GRU残差网络
 * 功能: 批量验证 data_csv 文件夹中的所有数据集
 * 
 * 使用方法:
 *   validate.exe
 * 
 * 输入:
 *   自动读取 ../python_workspace/data_csv/ 中所有 .csv 文件
 * 
 * 输出:
 *   结果保存到 ../python_workspace/results/YYYYMMDD_HHMMSS/ 文件夹
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifdef _WIN32
    #include <direct.h>
    #include <windows.h>
    #define mkdir(path, mode) _mkdir(path)
#endif
#include "neural_network.h"

#define MAX_FILES 100
#define MAX_PATH_LEN 512

/* ==================== 辅助函数 ==================== */

/**
 * 创建目录（递归创建）
 */
int create_directory(const char *path) {
    struct stat st = {0};
    if (stat(path, &st) == -1) {
        if (mkdir(path, 0755) != 0) {
            return -1;
        }
    }
    return 0;
}

/**
 * 获取当前时间戳字符串
 */
void get_timestamp(char *buffer, size_t size) {
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    strftime(buffer, size, "%Y%m%d_%H%M%S", t);
}

/**
 * 检查文件是否以指定后缀结尾
 */
int ends_with(const char *str, const char *suffix) {
    size_t str_len = strlen(str);
    size_t suffix_len = strlen(suffix);
    if (suffix_len > str_len) return 0;
    return strcmp(str + str_len - suffix_len, suffix) == 0;
}

/**
 * 列出目录中所有CSV文件
 */
int list_csv_files(const char *dir_path, char files[][MAX_PATH_LEN], int max_files) {
    int count = 0;
    char search_path[MAX_PATH_LEN];
    snprintf(search_path, sizeof(search_path), "%s\\*.csv", dir_path);
    
    #ifdef _WIN32
    WIN32_FIND_DATAA find_data;
    HANDLE hFind = FindFirstFileA(search_path, &find_data);
    
    if (hFind == INVALID_HANDLE_VALUE) {
        return 0;
    }
    
    do {
        if (!(find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            if (count < max_files) {
                strncpy(files[count], find_data.cFileName, MAX_PATH_LEN - 1);
                files[count][MAX_PATH_LEN - 1] = '\0';
                count++;
            }
        }
    } while (FindNextFileA(hFind, &find_data) != 0 && count < max_files);
    
    FindClose(hFind);
    #endif
    
    return count;
}

/* ==================== 数据读取函数 ==================== */

/**
 * 从CSV文件读取测试数据
 * @param filename CSV文件路径
 * @param samples 输出: 样本数据数组指针
 * @param num_samples 输出: 样本数量
 * @return 0成功, -1失败
 */
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

/* ==================== 结果保存函数 ==================== */

/**
 * 保存验证结果到CSV文件
 * @param filename 输出文件路径
 * @param samples 原始样本数据
 * @param predictions 模型预测值数组
 * @param num_samples 样本数量
 * @return 0成功, -1失败
 */
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

/* ==================== 打印函数 ==================== */

/**
 * 打印模型架构
 */
void print_model_architecture(void) {
    printf("Model Architecture:\n");
    printf("  +------------------------------------------+\n");
    printf("  |  Input Layer (6 features)                |\n");
    printf("  |  - x, dx, iq, diq, F0, mode_flag         |\n");
    printf("  +------------------------------------------+\n");
    printf("                   |\n");
    printf("                   v\n");
    printf("  +------------------------------------------+\n");
    printf("  |  PI Baseline Layer                       |\n");
    printf("  |  - %d Play operators                     |\n", NUM_PLAY_OPERATORS);
    printf("  |  - Learned threshold parameters          |\n");
    printf("  +------------------------------------------+\n");
    printf("                   |\n");
    printf("                   v\n");
    printf("  +------------------------------------------+\n");
    printf("  |  GRU Residual Network                    |\n");
    printf("  |  - Sequence Length: %-3d                  |\n", SEQ_LEN);
    printf("  |  - Hidden Size: %-3d                      |\n", HIDDEN_SIZE);
    printf("  |  - Learns force residual                 |\n");
    printf("  +------------------------------------------+\n");
    printf("                   |\n");
    printf("                   v\n");
    printf("  +------------------------------------------+\n");
    printf("  |  Output: Force Prediction                |\n");
    printf("  |  F_final = F_baseline + F_residual       |\n");
    printf("  +------------------------------------------+\n\n");
}

/**
 * 打印模型配置信息
 */
void print_model_config(void) {
    printf("Model Configuration:\n");
    printf("  +-------------------------------------+\n");
    printf("  | Network Parameters                  |\n");
    printf("  +-------------------------------------+\n");
    printf("  | Input Size      : %-3d               |\n", INPUT_SIZE);
    printf("  | Hidden Size     : %-3d               |\n", HIDDEN_SIZE);
    printf("  | Sequence Length : %-3d               |\n", SEQ_LEN);
    printf("  | PI Operators    : %-3d               |\n", NUM_PLAY_OPERATORS);
    printf("  +-------------------------------------+\n\n");
    
    printf("  +-------------------------------------+\n");
    printf("  | Sampling & Processing               |\n");
    printf("  +-------------------------------------+\n");
    printf("  | Sampling Period : %.4f s          |\n", DT);
    printf("  | Sampling Rate   : %.0f Hz          |\n", 1.0/DT);
    printf("  +-------------------------------------+\n\n");
    
    printf("  +-------------------------------------+\n");
    printf("  | Physical Constraints                |\n");
    printf("  +-------------------------------------+\n");
    if (APPLY_PHYSICAL_CONSTRAINT) {
        printf("  | Status          : Enabled           |\n");
        printf("  | Max Force Rate  : %.1f N/s         |\n", FORCE_MAX_RATE);
        printf("  | Min Force       : %.1f N           |\n", FORCE_MIN);
    } else {
        printf("  | Status          : Disabled          |\n");
    }
    printf("  +-------------------------------------+\n\n");
}

/**
 * 打印验证指标
 */
void print_validation_metrics(int num_samples, float mae, float rmse, float r2,
                              float min_true, float max_true, 
                              float min_pred, float max_pred) {
    printf("+============================================+\n");
    printf("|         Validation Results                 |\n");
    printf("+============================================+\n");
    printf("| Samples Tested  : %-8d                |\n", num_samples);
    printf("+--------------------------------------------+\n");
    printf("| Performance Metrics:                       |\n");
    printf("|   MAE  (Mean Absolute Error)   : %6.3f N  |\n", mae);
    printf("|   RMSE (Root Mean Square Error): %6.3f N  |\n", rmse);
    printf("|   R2   (Coefficient of Determ.): %6.4f    |\n", r2);
    printf("+--------------------------------------------+\n");
    printf("| Force Range Statistics:                    |\n");
    printf("|   Ground Truth: [%6.2f, %6.2f] N        |\n", min_true, max_true);
    printf("|   Predictions : [%6.2f, %6.2f] N        |\n", min_pred, max_pred);
    printf("+============================================+\n\n");
    
    // 性能评级
    printf("Performance Rating: ");
    if (r2 >= 0.98 && mae < 3.0f) {
        printf("***** Excellent\n");
    } else if (r2 >= 0.95 && mae < 5.0f) {
        printf("**** Very Good\n");
    } else if (r2 >= 0.90 && mae < 8.0f) {
        printf("*** Good\n");
    } else if (r2 >= 0.85) {
        printf("** Fair\n");
    } else {
        printf("* Needs Improvement\n");
    }
    printf("\n");
}

/* ==================== 主验证流程 ==================== */

int main(void) {
    printf("==================================================\n");
    printf("   Batch Validation - PI-GRU Neural Network\n");
    printf("==================================================\n\n");
    
    // 定义路径
    const char *data_csv_dir = "..\\python_workspace\\data_csv";
    const char *results_base_dir = "..\\python_workspace\\results";
    
    // 创建时间戳文件夹
    char timestamp[32];
    get_timestamp(timestamp, sizeof(timestamp));
    
    char results_dir[MAX_PATH_LEN];
    snprintf(results_dir, sizeof(results_dir), "%s\\%s", results_base_dir, timestamp);
    
    printf("Creating results directory: %s\n", results_dir);
    if (create_directory(results_dir) != 0) {
        printf("Error: Failed to create directory %s\n", results_dir);
        printf("Trying to create parent directory first...\n");
        create_directory(results_base_dir);
        if (create_directory(results_dir) != 0) {
            printf("Error: Still failed to create results directory\n");
            return 1;
        }
    }
    printf("[OK] Results directory created\n\n");
    
    // 列出所有CSV文件
    char csv_files[MAX_FILES][MAX_PATH_LEN];
    int num_files = list_csv_files(data_csv_dir, csv_files, MAX_FILES);
    
    if (num_files == 0) {
        printf("Error: No CSV files found in %s\n", data_csv_dir);
        return 1;
    }
    
    printf("Found %d CSV file(s) in data_csv folder:\n", num_files);
    for (int i = 0; i < num_files; i++) {
        printf("  [%d] %s\n", i + 1, csv_files[i]);
    }
    printf("\n");
    
    // 打印模型架构和配置（只打印一次）
    print_model_architecture();
    print_model_config();
    
    // 创建汇总结果文件
    char summary_path[MAX_PATH_LEN];
    snprintf(summary_path, sizeof(summary_path), "%s\\summary.txt", results_dir);
    FILE *summary_fp = fopen(summary_path, "w");
    if (summary_fp) {
        fprintf(summary_fp, "Batch Validation Summary\n");
        fprintf(summary_fp, "Timestamp: %s\n", timestamp);
        fprintf(summary_fp, "Total Files: %d\n\n", "File", "MAE", "RMSE", "R2");
        fprintf(summary_fp, "%-20s %10s %10s %10s\n", "File", "MAE", "RMSE", "R2");
        fprintf(summary_fp, "------------------------------------------------\n");
    }
    
    // 逐个验证文件
    int window_size = 5;
    for (int file_idx = 0; file_idx < num_files; file_idx++) {
        printf("==================================================\n");
        printf("Processing File %d/%d: %s\n", file_idx + 1, num_files, csv_files[file_idx]);
        printf("==================================================\n\n");
        
        // 构建完整路径
        char input_path[MAX_PATH_LEN];
        snprintf(input_path, sizeof(input_path), "%s\\%s", data_csv_dir, csv_files[file_idx]);
        
        // 构建输出文件名（去掉.csv后缀）
        char basename[MAX_PATH_LEN];
        strncpy(basename, csv_files[file_idx], MAX_PATH_LEN);
        char *dot = strrchr(basename, '.');
        if (dot) *dot = '\0';
        
        char output_path[MAX_PATH_LEN];
        snprintf(output_path, sizeof(output_path), "%s\\validation_%s.csv", results_dir, basename);
        
        // 读取数据
        SampleData *samples = NULL;
        int num_samples = 0;
        
        printf("Loading data...\n");
        if (read_csv_data(input_path, &samples, &num_samples) != 0) {
            printf("[WARNING] Skipping file due to read error\n\n");
            if (summary_fp) {
                fprintf(summary_fp, "%-20s %10s %10s %10s\n", csv_files[file_idx], "ERROR", "ERROR", "ERROR");
            }
            continue;
        }
        
        // 分配预测结果数组
        float *predictions = (float *)malloc(num_samples * sizeof(float));
        if (predictions == NULL) {
            printf("Error: Memory allocation failed\n");
            free(samples);
            continue;
        }
        
        // 执行推理
        printf("Running inference on %d samples...\n", num_samples);
        model_inference(samples, num_samples, predictions, window_size);
        printf("[OK] Inference completed\n\n");
        
        // 计算指标
        float *ground_truth = (float *)malloc(num_samples * sizeof(float));
        for (int i = 0; i < num_samples; i++) {
            ground_truth[i] = samples[i].F;
        }
        
        float mae, rmse, r2;
        compute_metrics(predictions, ground_truth, num_samples, &mae, &rmse, &r2);
        
        // 计算统计信息
        float min_pred = predictions[0], max_pred = predictions[0];
        float min_true = ground_truth[0], max_true = ground_truth[0];
        for (int i = 1; i < num_samples; i++) {
            if (predictions[i] < min_pred) min_pred = predictions[i];
            if (predictions[i] > max_pred) max_pred = predictions[i];
            if (ground_truth[i] < min_true) min_true = ground_truth[i];
            if (ground_truth[i] > max_true) max_true = ground_truth[i];
        }
        
        // 打印结果
        print_validation_metrics(num_samples, mae, rmse, r2, min_true, max_true, min_pred, max_pred);
        
        // 保存结果
        printf("Saving results to: %s\n", output_path);
        save_results_csv(output_path, samples, predictions, num_samples);
        printf("[OK] Results saved\n\n");
        
        // 写入汇总文件
        if (summary_fp) {
            fprintf(summary_fp, "%-20s %10.3f %10.3f %10.4f\n", 
                    csv_files[file_idx], mae, rmse, r2);
        }
        
        // 释放内存
        free(samples);
        free(predictions);
        free(ground_truth);
    }
    
    // 关闭汇总文件
    if (summary_fp) {
        fclose(summary_fp);
        printf("Summary saved to: %s\n\n", summary_path);
    }
    
    printf("==================================================\n");
    printf("  Batch Validation Completed!\n");
    printf("  Processed: %d file(s)\n", num_files);
    printf("  Results saved in: %s\n", results_dir);
    printf("==================================================\n\n");
    
    return 0;
}
