# 加载所需库 
library(tidyverse) 

library(caret) 
library(e1071) 
library(xgboost) 
library(randomForest) 
library(ggplot2) 
library(ggcorrplot)  
library(gridExtra) 
library(viridis) 
library(patchwork) 
library(showtext) 

showtext_auto() 


# 设置中文显示 
if (.Platform$OS.type == "windows") { 
  windowsFonts( 
    SimHei = windowsFont("SimHei"), 
    Song = windowsFont("SimSun"), 
    Kai = windowsFont("KaiTi") 
  ) 
} else { 
  # 非Windows系统可能需要其他字体设置 
  theme_update(text = element_text(family = "Arial Unicode MS")) 
} 

# 设置随机种子，确保结果可复现 
set.seed(123) 

# 1. 数据读取与预处理 
# 假设训练集和测试集分别保存为train_data.csv和test_data.csv 
train_data <- read.csv("/Users/slh/Downloads/planB/train_features.csv", stringsAsFactors = FALSE) 
test_data <- read.csv("/Users/slh/Downloads/planB/test_features.csv", stringsAsFactors = FALSE) 

# 查看数据结构 
str(train_data) 
str(test_data) 

# 提取特征和目标变量 
# 将fault_type作为目标变量 
y_train <- factor(train_data$fault_class) 
y_test <- factor(test_data$fault_class) 

# 除去filename、fault_size、load和fault_type外的变量作为特征 
exclude_cols <- c("file_path", "fault_class") 
X_train <- train_data[, !(colnames(train_data) %in% exclude_cols)] 
X_test <- test_data[, !(colnames(test_data) %in% exclude_cols)] 

# 检查是否有缺失值 
sum(is.na(X_train)) 
sum(is.na(X_test)) 

# 如果有缺失值，进行简单填充（根据实际情况调整） 
if (sum(is.na(X_train)) > 0) { 
  preProcess_missingdata <- preProcess(X_train, method = "medianImpute") 
  X_train <- predict(preProcess_missingdata, newdata = X_train) 
  X_test <- predict(preProcess_missingdata, newdata = X_test) 
} 

# 特征标准化 
preProcess_range_model <- preProcess(X_train, method = "range") 
X_train_scaled <- predict(preProcess_range_model, newdata = X_train) 
X_test_scaled <- predict(preProcess_range_model, newdata = X_test) 

# 2. 支持向量机(SVM)模型 
# 训练SVM模型 
svm_model <- svm(x = X_train_scaled, y = y_train, kernel = "radial", cost = 1, gamma = 0.1) 

# 在测试集上预测 
svm_pred <- predict(svm_model, X_test_scaled) 

# 计算准确率 
svm_acc <- mean(svm_pred == y_test) 

cat("SVM模型测试集准确率:", svm_acc, "\n") 

# 3. XGBoost模型 
# 数据转换为xgb矩阵格式 
dtrain <- xgb.DMatrix(data = as.matrix(X_train_scaled), label = as.numeric(y_train) - 1) 
dtest <- xgb.DMatrix(data = as.matrix(X_test_scaled), label = as.numeric(y_test) - 1) 

# 设置5折交叉验证 
cv_folds <- createFolds(y_train, k = 5, returnTrain = TRUE) 

# 定义超参数搜索范围 
param_grid <- expand.grid( 
  nrounds = c(100, 200, 300), 
  max_depth = c(3, 5, 7), 
  eta = c(0.01, 0.1, 0.3), 
  subsample = c(0.7, 0.8, 0.9), 
  colsample_bytree = c(0.7, 0.8, 0.9), 
  min_child_weight = 1, 
  gamma = 0, 
  lambda = 1, 
  alpha = 0 
) 

# 存储交叉验证结果 
cv_results <- data.frame() 

# 超参数搜索 
for (i in 1:nrow(param_grid)) { 
  params <- list( 
    objective = "multi:softmax", 
    num_class = length(levels(y_train)), 
    max_depth = param_grid$max_depth[i], 
    eta = param_grid$eta[i], 
    subsample = param_grid$subsample[i], 
    colsample_bytree = param_grid$colsample_bytree[i], 
    min_child_weight = param_grid$min_child_weight[i], 
    gamma = param_grid$gamma[i], 
    lambda = param_grid$lambda[i], 
    alpha = param_grid$alpha[i] 
  ) 

  # 5折交叉验证 
  xgb_cv <- xgb.cv( 
    params = params, 
    data = dtrain, 
    nrounds = param_grid$nrounds[i], 
    folds = cv_folds, 
    metrics = "merror", 
    early_stopping_rounds = 50, 
    verbose = 0 
  ) 

  # 保存结果 
  cv_results <- rbind(cv_results, data.frame( 
    nrounds = param_grid$nrounds[i], 
    max_depth = param_grid$max_depth[i], 
    eta = param_grid$eta[i], 
    subsample = param_grid$subsample[i], 
    colsample_bytree = param_grid$colsample_bytree[i], 
    mean_train_error = mean(xgb_cv$evaluation_log$train_merror_mean), 
    mean_test_error = mean(xgb_cv$evaluation_log$test_merror_mean) 
  )) 

  # 打印进度 
  if (i %% 10 == 0) { 
    cat("完成", i, "/", nrow(param_grid), "轮超参数搜索\n") 
  } 
} 

# 找出最佳参数（测试误差最小） 
best_params_idx <- which.min(cv_results$mean_test_error) 
best_params <- param_grid[best_params_idx, ] 
cat("XGBoost最佳参数:\n") 
print(best_params) 


# 超参数搜索结果可视化 
cv_results_long <- cv_results %>% 
  select(-mean_train_error) %>% 
  pivot_longer(cols = mean_test_error, names_to = "type", values_to = "error") 

# 学习率与误差关系 
p1 <- ggplot(cv_results, aes(x = as.factor(eta), y = mean_test_error)) + 
  geom_boxplot(fill = "#4285f4") + 
  labs(x = "学习率 (eta)", y = "平均测试误差", title = "学习率对模型性能的影响") + 
  theme_minimal() + 
  theme(plot.title = element_text(hjust = 0.5)) 

# 树深度与误差关系 
p2 <- ggplot(cv_results, aes(x = as.factor(max_depth), y = mean_test_error)) + 
  geom_boxplot(fill = "#ea4335") + 
  labs(x = "树的最大深度", y = "平均测试误差", title = "树深度对模型性能的影响") + 
  theme_minimal() + 
  theme(plot.title = element_text(hjust = 0.5)) 

# 子样本比例与误差关系 
p3 <- ggplot(cv_results, aes(x = as.factor(subsample), y = mean_test_error)) + 
  geom_boxplot(fill = "#fbbc05") + 
  labs(x = "子样本比例", y = "平均测试误差", title = "子样本比例对模型性能的影响") + 
  theme_minimal() + 
  theme(plot.title = element_text(hjust = 0.5)) 

# 特征采样比例与误差关系 
p4 <- ggplot(cv_results, aes(x = as.factor(colsample_bytree), y = mean_test_error)) + 
  geom_boxplot(fill = "#34a853") + 
  labs(x = "特征采样比例", y = "平均测试误差", title = "特征采样比例对模型性能的影响") + 
  theme_minimal() + 
  theme(plot.title = element_text(hjust = 0.5)) 

# 组合图形 
grid.arrange(p1, p2, p3, p4, ncol = 2) 
ggsave("xgb_hyperparameter_impact.png", width = 12, height = 10, dpi = 300) 

# 使用最佳参数在整个训练集上重新训练模型 
xgb_final_params <- list( 
  objective = "multi:softmax", 
  num_class = length(levels(y_train)), 
  max_depth = best_params$max_depth, 
  eta = best_params$eta, 
  subsample = best_params$subsample, 
  colsample_bytree = best_params$colsample_bytree, 
  min_child_weight = 1, 
  gamma = 0, 
  lambda = 1, 
  alpha = 0 
) 

xgb_model <- xgb.train( 
  params = xgb_final_params, 
  data = dtrain, 
  nrounds = best_params$nrounds, 
  verbose = 0 
) 

# 在测试集上预测 
xgb_pred <- predict(xgb_model, dtest) 
xgb_pred <- factor(xgb_pred, levels = 0:(length(levels(y_train))-1), labels = levels(y_train)) 

# 计算准确率 
xgb_acc <- mean(xgb_pred == y_test) 
cat("XGBoost模型测试集准确率:", xgb_acc, "\n") 

# 4. 随机森林模型 
# 训练随机森林模型 
rf_model <- randomForest( 
  x = X_train_scaled, 
  y = y_train, 
  ntree = 200, 

  mtry = sqrt(ncol(X_train_scaled)), 
  importance = TRUE, 
  proximity = TRUE 
) 

# 模型训练过程可视化（误差随树数量的变化） 
plot(rf_model, main = "随机森林误差随树数量的变化") 
legend("topright", colnames(rf_model$err.rate), col = 1:3, fill = 1:3) 
dev.copy(png, "rf_error_by_trees.png", width = 800, height = 600) 
dev.off() 

# 在测试集上预测 
rf_pred <- predict(rf_model, X_test_scaled) 

# 计算准确率 
rf_acc <- mean(rf_pred == y_test) 
cat("随机森林模型测试集准确率:", rf_acc, "\n") 

# 5. 模型性能评估与比较 
# 存储所有模型的预测结果 
all_preds <- data.frame( 
  actual = y_test, 
  svm = svm_pred, 
  xgb = xgb_pred, 
  rf = rf_pred 
) 

# 计算各模型的评估指标 
calculate_metrics <- function(actual, pred) { 
  cm <- confusionMatrix(pred, actual) 
  data.frame( 
    accuracy = cm$overall["Accuracy"], 
    precision = mean(cm$byClass[, "Precision"], na.rm = TRUE), 
    recall = mean(cm$byClass[, "Recall"], na.rm = TRUE), 
    f1 = mean(cm$byClass[, "F1"], na.rm = TRUE) 
  ) 
} 

# 计算每个模型的评估指标 
metrics <- rbind( 
  calculate_metrics(all_preds$actual, all_preds$svm) %>% mutate(model = "SVM"), 
  calculate_metrics(all_preds$actual, all_preds$xgb) %>% mutate(model = "XGBoost"), 
  calculate_metrics(all_preds$actual, all_preds$rf) %>% mutate(model = "随机森林") 
) 

# 重新排列列顺序 
metrics <- metrics %>% select(model, accuracy, precision, recall, f1) 
print(metrics) 

# 可视化各模型性能指标 
metrics_long <- metrics %>% 
  pivot_longer(cols = -model, names_to = "metric", values_to = "value") %>% 
  mutate(metric = factor(metric, levels = c("accuracy", "precision", "recall", "f1"), 
                         labels = c("准确率", "精确率", "召回率", "F1分数"))) 

ggplot(metrics_long, aes(x = model, y = value, fill = model)) + 
  geom_bar(stat = "identity", position = "dodge") + 
  facet_wrap(~metric) + 
  scale_fill_viridis(discrete = TRUE) + 
  labs(title = "不同模型的性能指标比较", x = "模型", y = "分数") + 
  theme_minimal() + 
  theme( 
    plot.title = element_text(hjust = 0.5, size = 14), 
    axis.text.x = element_text(angle = 45, hjust = 1) 
  ) 
ggsave("model_metrics_comparison.png", width = 12, height = 8, dpi = 300) 

# 6. 混淆矩阵可视化 
plot_confusion_matrix <- function(actual, pred, model_name) { 
  cm <- confusionMatrix(pred, actual) 
  cm_df <- as.data.frame(cm$table) 

  ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) + 

    geom_tile(color = "white") + 
    geom_text(aes(label = Freq), vjust = 0.5, size = 4) + 
    scale_fill_viridis(option = "plasma") + 
    labs(title = paste(model_name, "模型混淆矩阵"), x = "实际类别", y = "预测类别") + 
    theme_minimal() + 
    theme( 
      plot.title = element_text(hjust = 0.5, size = 14), 
      axis.text.x = element_text(angle = 45, hjust = 1) 
    ) 
} 

# 绘制三个模型的混淆矩阵 
p_svm <- plot_confusion_matrix(all_preds$actual, all_preds$svm, "SVM") 
p_xgb <- plot_confusion_matrix(all_preds$actual, all_preds$xgb, "XGBoost") 
p_rf <- plot_confusion_matrix(all_preds$actual, all_preds$rf, "随机森林") 

# 组合混淆矩阵图 
(p_svm + p_xgb + p_rf) + plot_annotation(title = "各模型混淆矩阵比较") 
ggsave("confusion_matrices.png", width = 18, height = 6, dpi = 300) 

# 7. 特征重要性可视化 
# 提取XGBoost特征重要性 
xgb_imp <- xgb.importance(feature_names = colnames(X_train_scaled), model = xgb_model) 
xgb_imp_df <- as.data.frame(xgb_imp) %>% 
  mutate(Feature = factor(Feature, levels = rev(Feature))) %>% 
  head(15)  # 取前15个重要特征 

# 提取随机森林特征重要性 
rf_imp <- as.data.frame(importance(rf_model)) %>% 
  mutate(Feature = rownames(.)) %>% 
  rename(Importance = MeanDecreaseGini) %>% 
  arrange(desc(Importance)) %>% 
  head(15) %>%  # 取前15个重要特征 
  mutate(Feature = factor(Feature, levels = rev(Feature))) 

# 绘制XGBoost特征重要性 
p_xgb_imp <- ggplot(xgb_imp_df, aes(x = Feature, y = Gain)) + 
  geom_bar(stat = "identity", fill = "#4285f4") + 
  coord_flip() + 
  labs(title = "XGBoost模型特征重要性", x = "特征", y = "增益") + 
  theme_minimal() + 
  theme(plot.title = element_text(hjust = 0.5)) 

# 绘制随机森林特征重要性 
p_rf_imp <- ggplot(rf_imp, aes(x = Feature, y = Importance)) + 
  geom_bar(stat = "identity", fill = "#34a853") + 
  coord_flip() + 
  labs(title = "随机森林模型特征重要性", x = "特征", y = "不纯度减少量") + 
  theme_minimal() + 
  theme(plot.title = element_text(hjust = 0.5)) 

# 组合特征重要性图 
grid.arrange(p_xgb_imp, p_rf_imp, ncol = 2) 
ggsave("feature_importance.png", width = 14, height = 8, dpi = 300) 

# 输出最终评估结果 
cat("\n模型性能总结:\n") 
print(metrics)