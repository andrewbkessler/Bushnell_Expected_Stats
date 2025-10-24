suppressPackageStartupMessages(library(tidyverse))
library(baseballr)
library(caTools)
library(class)
library(caret)
library(gbm)
library(xgboost)
library(Matrix)
library(vip)
options(scipen = 999)

### MLB Data
# mlb_schedule <- mlb_schedule(season = 2025) |> 
#   filter(as.Date(date) >= '2025-09-01' & as.Date(date) < '2025-10-01' & status_status_code=='F' & game_type=='R')
# mlb_pbp <- data.frame()
# for(i in 1:nrow(mlb_schedule)){
#   temp_pbp <- mlb_pbp(mlb_schedule$game_pk[i])
#   mlb_pbp <- rbind(mlb_pbp, temp_pbp, fill = TRUE)
# }
# write_csv(mlb_pbp, "model_data/mlb_pbp.csv")
mlb_pbp <- read.csv("model_data/mlb_pbp.csv")

### MLB At Bats
mlb_plate_aps <- mlb_pbp |> 
  group_by(game_pk, atBatIndex) |> 
  slice_max(order_by = pitchNumber, n = 1) |> ungroup()
mlb_batted_balls <- mlb_plate_aps |> 
  filter(details.isInPlay==TRUE & !is.na(hitData.launchAngle) & !is.na(hitData.launchSpeed)) |> 
  mutate(
    total_bases = case_when(
      result.eventType == 'single' ~ 1,
      result.eventType == 'double' ~ 2,
      result.eventType == 'triple' ~ 3,
      result.eventType == 'home_run' ~ 4,
      TRUE ~ 0),
    hit = ifelse(total_bases > 0, 1, 0)) |> 
  select(game_pk, atBatIndex, matchup.batter.id, matchup.batter.fullName, hitData.launchSpeed, hitData.launchAngle, total_bases)

### Prepare Model Data
join_model_data <- mlb_batted_balls
model_data <- join_model_data |> 
  select(-game_pk:-matchup.batter.fullName)



##### Note: I originally used the Class method for KNN, but found the Carot version better
# ################### KNN - Class Method ##################
# ### Split into train and test
# set.seed(2018)
# split <- sample.split(model_data$total_bases, 
#                      SplitRatio = 0.75)
# train <- subset(model_data, split == TRUE)
# test <- subset(model_data, split == FALSE)
# 
# ### Feature Scaling
# train_scaled <- scale(train[-3])
# test_scaled <- scale(test[-3])
# 
# ### Train Model
# test_pred <- knn(
#   train = train_scaled, 
#   test = test_scaled,
#   cl = train$total_bases, 
#   k=11)
# 
# ### Model Evaluation
# actual <- test$total_bases
# cm <- table(actual,test_pred)
# cm
# accuracy <- sum(diag(cm))/length(actual)
# sprintf("Accuracy: %.2f%%", accuracy*100)




################### KNN - Carot Method ##################
### Split into train and test
set.seed(2018)
knn_model_data <- model_data
knn_model_data$total_bases <- factor(knn_model_data$total_bases, levels = c(0, 1, 2, 3, 4))
knn_trainIndex <- createDataPartition(knn_model_data$total_bases, 
                                  times=1, 
                                  p = .8, 
                                  list = FALSE)
knn_train <- knn_model_data[knn_trainIndex, ]
knn_test_merged <- join_model_data[-knn_trainIndex, ]
knn_test <- knn_model_data[-knn_trainIndex, ]

### Data Preprocessing (scaling)
knn_preProcValues <- preProcess(knn_train, method = c("center", "scale"))
knn_trainTransformed <- predict(knn_preProcValues, knn_train)
knn_testTransformed <- predict(knn_preProcValues, knn_test)

### Model Tuning (finding optimal K value)
knnModel <- train(
  total_bases ~ ., 
  data = knn_trainTransformed, 
  method = "knn", 
  trControl = trainControl(method = "cv"), 
  tuneGrid = data.frame(k = c(3,5,7,9,11,13,15,17,19,21,23,25,27,29,31))
)
knnModel$bestTune$k

### Training best model
knn_best_model<- knn3(
  total_bases ~ .,
  data = knn_trainTransformed,
  k = knnModel$bestTune$k
)

### Model Evaluation (class version)
knn_predictions <- predict(knn_best_model, knn_testTransformed,type = "class")
### Calculate confusion matrix
knn_cm <- confusionMatrix(knn_predictions, knn_testTransformed$total_bases)
knn_cm
knn_actual <- knn_test$total_bases
knn_cm_accuracy <- table(knn_actual,knn_predictions)
knn_accuracy <- sum(diag(knn_cm_accuracy))/length(knn_actual)
sprintf("Accuracy: %.2f%%", knn_accuracy*100)

### Model Evaluation (probability version)
knn_probs <- predict(knn_best_model, knn_testTransformed,type = "prob")
head(knn_probs, 20)
knn_merged_probabilities <- cbind(knn_test_merged, knn_probs) |> 
  rename('prob_0'='0', 'prob_1'='1', 'prob_2'='2', 'prob_3'='3', 'prob_4'='4') |> 
  mutate(exp_bases = prob_1*1 + prob_2*2 + prob_3*3 + prob_4*4)

### Saving Model Components
save_train_merged <- join_model_data
save_train_data <- knn_model_data
save_train_labels <- knn_model_data[, 3]
k_value <- knnModel$bestTune$k

knn_components <- list(
  train_merged = save_train_merged,
  train_data = save_train_data,
  train_labels = save_train_labels,
  k = k_value
)
saveRDS(knn_components, file = "model_data/knn_setup.Rds")



################### XGBoost ##################
### Split into Test/Train
set.seed(2018)
train_index <- sample(1:nrow(model_data), 0.7 * nrow(model_data))
train_data <- model_data[train_index, ]
test_merged <- join_model_data[-train_index, ]
test_data <- model_data[-train_index, ]

### Split Train into a subsample and Validation dataset
subsample_index <- sample(1:nrow(test_data), 0.7 * nrow(test_data))
test_merged <- test_merged[subsample_index, ]
test_data <- test_data[subsample_index, ]
valid_merged <- test_merged[-subsample_index, ]
valid_data <- test_data[-subsample_index, ]


### Create Feature Matrix and Label Vector
train_x <- as.matrix(train_data[, 1:2])
train_y <- train_data$total_bases
test_x <- as.matrix(test_data[, 1:2])
test_y <- test_data$total_bases
valid_x <- as.matrix(valid_data[, 1:2])
valid_y <- valid_data$total_bases

### Convert train & validation to DMatrix
dtrain <- xgb.DMatrix(data = train_x, label = train_y)
dvalid <- xgb.DMatrix(data = valid_x, label = valid_y)

### Set Parameters
# params <- list(
#   objective = "multi:softprob",  # returns probabilities for each class
#   num_class = 5,                 # total_bases ranges from 0 to 4
#   eval_metric = "mlogloss")

### Train the Model
watchlist <- list(train = dtrain, valid = dvalid)
xgboost_model <- xgb.train(
  data = dtrain,
  objective = "multi:softprob",
  num_class = 5,
  eval_metric = "mlogloss",
  watchlist = watchlist,
  nrounds = 1000,
  early_stopping_rounds = 50,
  print_every_n = 100,
  verbose = 1
)

### Make Predictions
pred_probs <- predict(xgboost_model, test_x)  # returns a matrix of probabilities
pred_matrix <- matrix(pred_probs, ncol = 5, byrow = TRUE)
merged_predictions <- cbind(test_merged, pred_matrix) |> 
  rename('prob_0'='1', 'prob_1'='2', 'prob_2'='3', 'prob_3'='4', 'prob_4'='5') |> 
  mutate(exp_bases = prob_1*1 + prob_2*2 + prob_3*3 + prob_4*4)

### Evaluate Model
vip(xgboost_model)
pred_class <- max.col(pred_matrix) - 1
accuracy <- mean(pred_class == test_y)
print(paste("Accuracy:", round(accuracy, 4)))
conf_matrix <- confusionMatrix(factor(pred_class), factor(test_y))
print(conf_matrix)
log_loss <- function(actual, predicted_probs, eps = 1e-15) {
  n <- length(actual)
  clipped_probs <- pmax(pmin(predicted_probs, 1 - eps), eps)
  idx <- cbind(1:n, actual + 1)  # +1 because R is 1-indexed
  loss <- -mean(log(clipped_probs[idx]))
  return(loss)
}

logloss <- log_loss(test_y, pred_matrix)
print(paste("Log Loss:", round(logloss, 4)))

### Save the model
xgb.save(xgboost_model, "model_data/xgboost_model.model")



################### Gradient Boosting Classifier ##################
### NOTE: There is some issue with the "multinomial" distribution, so I don't think GBC will work
# set.seed(2018)
# train_index <- sample(1:nrow(model_data), 0.7 * nrow(model_data))
# train_data <- model_data[train_index, ]
# test_data <- model_data[-train_index, ]
# 
# ### Train Model
# gbm_model <- gbm(
#   formula = total_bases ~ .,
#   distribution = "multinomial",
#   data = train_data,
#   n.trees = 500,
#   interaction.depth = 3,
#   shrinkage = 0.1,
#   cv.folds = 5,
#   verbose = FALSE
# )
# 
# ### Check for Errors (NA values)
# summary(gbm_model$cv.error)
# 
# ### Evaluate Model
# best_iter <- gbm.perf(gbm_model, method = "cv")
# pred_probs <- predict(gbm_model, test_data, n.trees = best_iter, type = "response")
# pred_classes <- colnames(pred_probs)[apply(pred_probs, 1, which.max)]
# confusion_matrix <- table(Predicted = pred_classes, Actual = test_data$total_bases)
# confusion_matrix
# accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
# print(accuracy)
# 
# ### Saving Model
# saveRDS(gbm_model, file = "model_data/gbm_model.Rds")








