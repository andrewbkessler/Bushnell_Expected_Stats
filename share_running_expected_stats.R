suppressPackageStartupMessages(library(tidyverse))
library(caret)
library(gbm)
library(xgboost)
library(vip)
options(scipen=999)

### I removed the code to pull the game data
### Let me know if you want this and I can find a better way to share

### Bushnell At Bats
bushnell_pas <- game_pbp |> 
  filter(BatterTeam=='BUS_BEA') |> 
  group_by(file_name, Inning, PAofInning) |> 
  slice_max(order_by = PitchofPA, n = 1) |> ungroup()
bushnell_abs <- bushnell_pas |> 
  filter(PitchCall=='InPlay' | KorBB=='Strikeout')
bushnell_batted_balls <- bushnell_abs |> 
  filter(PitchCall=='InPlay' & !is.na(ExitSpeed) & !is.na(Angle)) |> 
  mutate(
    total_bases = case_when(
      PlayResult == 'Single' ~ 1,
      PlayResult == 'Double' ~ 2,
      PlayResult == 'Triple' ~ 3,
      PlayResult == 'HomeRun' ~ 4,
      TRUE ~ 0)) |> 
  select(file_name, Inning, PAofInning, BatterId, Batter, ExitSpeed, Angle, total_bases)

### Prepare Bushnell Data
join_bushnell_data <- bushnell_batted_balls
bushnell_data <- join_bushnell_data |>
  select(-file_name:-Batter, -total_bases)

### Get Strikeouts and Walks for analysis
strikeouts_and_walks <- bushnell_pas |> 
  filter(KorBB%in%c('Strikeout','Walk')) |> 
  mutate(total_bases = ifelse(KorBB=='Strikeout', 0, 1),
         prob_0 = ifelse(KorBB=='Strikeout', 1, 0),
         prob_1 = ifelse(KorBB=='Strikeout', 0, 1),
         prob_2 = 0,
         prob_3 = 0,
         prob_4 = 0,
         exp_bases = ifelse(KorBB=='Strikeout', 0, 1),
         hit_prob = 0) |> 
  select(file_name, Inning, PAofInning, BatterId, Batter, ExitSpeed, Angle, total_bases:hit_prob, play_type=KorBB)





###################### KNN Model ######################
### Read Training Components (and rename variables)
loaded_knn_components <- readRDS("model_data/knn_setup.Rds")
loaded_train_data <- cbind(loaded_knn_components$train_data, loaded_knn_components$train_labels) |> 
  rename('ExitSpeed'='hitData.launchSpeed', 'Angle'='hitData.launchAngle')
loaded_k <- loaded_knn_components$k
rm(loaded_knn_components)

### Make Total Bases a Factor
loaded_train_data$total_bases <- factor(loaded_train_data$total_bases, levels = c(0, 1, 2, 3, 4))

### Data Preprocessing
preProcValues <- preProcess(loaded_train_data, method = c("center", "scale"))
loaded_train_dataTransformed <- predict(preProcValues, loaded_train_data)
bushnell_dataTransformed <- predict(preProcValues, bushnell_data)

### Model Tuning (finding optimal K value)
knnModel <- train(
  total_bases ~ ., 
  data = loaded_train_dataTransformed, 
  method = "knn", 
  trControl = trainControl(method = "cv"), 
  tuneGrid = data.frame(k = c(3,5,7,9,11,13,15,17,19,21,23,25,27,29,31))
)
knnModel$bestTune$k

### Training best model
best_model<- knn3(
  total_bases ~ .,
  data = loaded_train_dataTransformed,
  k = knnModel$bestTune$k
)

### Model Evaluation (probability version)
probs <- predict(best_model, bushnell_dataTransformed, type = "prob")
# head(probs, 20)
merged_probabilities <- cbind(join_bushnell_data, probs) |> 
  rename('prob_0'='0', 'prob_1'='1', 'prob_2'='2', 'prob_3'='3', 'prob_4'='4') |> 
  mutate(exp_bases = prob_1*1 + prob_2*2 + prob_3*3 + prob_4*4,
         hit_prob = prob_1 + prob_2 + prob_3 + prob_4,
         play_type = 'InPlay')
# rm(bushnell_data, bushnell_dataTransformed, join_bushnell_data, loaded_train_data, loaded_train_dataTransformed,
#    preProcValues, probs, loaded_k, best_model, knnModel)



###################### KNN Analysis ######################
pa_with_exp <- rbind(merged_probabilities, strikeouts_and_walks) |> 
  mutate(count = 1)

expected_metrics <- pa_with_exp |> 
  group_by(BatterId, Batter) |> 
  summarise(plate_apps = sum(count),
            at_bats = sum(count[play_type!='Walk']),
            hits = sum(count[play_type=='InPlay' & total_bases>0]),
            strikeouts = sum(count[play_type=='Strikeout']),
            inPlayOut = sum(count[play_type=='InPlay' & total_bases==0]),
            walks = sum(count[play_type=='Walk']),
            ba = sum(count[play_type=='InPlay' & total_bases>0])/at_bats,
            exp_ba = sum(hit_prob)/at_bats,
            obp = sum(count[total_bases>0])/plate_apps,
            exp_obp = (walks + sum(hit_prob))/plate_apps,
            slug = sum(total_bases[play_type=='InPlay'])/at_bats,
            exp_slug = sum(exp_bases[play_type=='InPlay'])/at_bats,
            ops = obp + slug,
            exp_ops = exp_obp + exp_slug) |> 
  mutate(ba_over_exp = ba-exp_ba,
         ops_over_exp = ops-exp_ops)

### Graphs
ba_over_exp_graph <- ggplot(expected_metrics) +
  geom_segment(aes(x=Batter, xend=Batter, y=ba, yend=exp_ba), color="grey") +
  geom_point(aes(x=Batter, y=ba), color="grey", size=4) +
  geom_point(aes(x=Batter, y=exp_ba), color="green", size=4) +
  theme_light() +
  labs(title = "Bushnell Batting Avg v Expected",
       subtitle = "Grey = Actual BA     |    Green = Expected BA",
       x = "Batter",
       y = "Batting Average") +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        panel.grid.major.x = element_blank(),
        panel.border = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))
ba_over_exp_graph
ops_over_exp_graph <- ggplot(expected_metrics) +
  geom_segment(aes(x=Batter, xend=Batter, y=ops, yend=exp_ops), color="grey") +
  geom_point(aes(x=Batter, y=ops), color="grey", size=4) +
  geom_point(aes(x=Batter, y=exp_ops), color="green", size=4) +
  theme_light() +
  labs(title = "Bushnell OPS v Expected",
       subtitle = "Grey = Actual OPS     |    Green = Expected OPS",
       x = "Batter",
       y = "OPS") +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        panel.grid.major.x = element_blank(),
        panel.border = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))
ops_over_exp_graph

### Hit Accuracy
knn_hit_accuracy <- merged_probabilities |> 
  mutate(hit = ifelse(total_bases > 0, 1, 0),
         ba_error = hit_prob - hit) |> 
  summarise(avg_error = mean(ba_error),
            abs_avg_error = mean(abs(ba_error)))







###################### XG Boost Model ######################
### Load model
xgboost_model <- xgb.load("model_data/xgboost_model.model")

### Prepare Bsuhnell Data
xg_bushnell_data_x <- as.matrix(bushnell_data)

### Run model on Bushnell Data
xg_pred_probs <- predict(xgboost_model, xg_bushnell_data_x)
xg_pred_matrix <- matrix(xg_pred_probs, ncol = 5, byrow = TRUE)
xg_merged_probabilities <- cbind(join_bushnell_data, xg_pred_matrix) |> 
  rename('prob_0'='1', 'prob_1'='2', 'prob_2'='3', 'prob_3'='4', 'prob_4'='5') |> 
  mutate(exp_bases = prob_1*1 + prob_2*2 + prob_3*3 + prob_4*4,
         hit_prob = prob_1 + prob_2 + prob_3 + prob_4,
         play_type = 'InPlay')

### Evaluate Model
xg_pred_class <- max.col(xg_pred_matrix) - 1
xg_accuracy <- mean(xg_pred_class == join_bushnell_data$total_bases)
print(paste("Accuracy:", round(xg_accuracy, 4)))
xg_conf_matrix <- confusionMatrix(factor(xg_pred_class), factor(join_bushnell_data$total_bases))
print(xg_conf_matrix)
log_loss_func <- function(actual, predicted_probs, eps = 1e-15) {
  n <- length(actual)
  clipped_probs <- pmax(pmin(predicted_probs, 1 - eps), eps)
  idx <- cbind(1:n, actual + 1)  # +1 because R is 1-indexed
  loss <- -mean(log(clipped_probs[idx]))
  return(loss)
}

xg_logloss <- log_loss_func(join_bushnell_data$total_bases, xg_pred_matrix)
print(paste("Log Loss:", round(xg_logloss, 4)))


###################### XGBoost Analysis ######################
xg_pa_with_exp <- rbind(xg_merged_probabilities, strikeouts_and_walks) |> 
  mutate(count = 1)

xg_expected_metrics <- xg_pa_with_exp |> 
  group_by(BatterId, Batter) |> 
  summarise(plate_apps = sum(count),
            at_bats = sum(count[play_type!='Walk']),
            hits = sum(count[play_type=='InPlay' & total_bases>0]),
            strikeouts = sum(count[play_type=='Strikeout']),
            inPlayOut = sum(count[play_type=='InPlay' & total_bases==0]),
            walks = sum(count[play_type=='Walk']),
            ba = sum(count[play_type=='InPlay' & total_bases>0])/at_bats,
            exp_ba = sum(hit_prob)/at_bats,
            obp = sum(count[total_bases>0])/plate_apps,
            exp_obp = (walks + sum(hit_prob))/plate_apps,
            slug = sum(total_bases[play_type=='InPlay'])/at_bats,
            exp_slug = sum(exp_bases[play_type=='InPlay'])/at_bats,
            ops = obp + slug,
            exp_ops = exp_obp + exp_slug) |> 
  mutate(ba_over_exp = ba-exp_ba,
         ops_over_exp = ops-exp_ops)

### Graphs
ba_over_exp_graph <- ggplot(xg_expected_metrics) +
  geom_segment(aes(x=Batter, xend=Batter, y=ba, yend=exp_ba), color="grey") +
  geom_point(aes(x=Batter, y=ba), color="grey", size=4) +
  geom_point(aes(x=Batter, y=exp_ba), color="green", size=4) +
  theme_light() +
  labs(title = "Bushnell Batting Avg v Expected",
       subtitle = "Grey = Actual BA     |    Green = Expected BA",
       x = "Batter",
       y = "Batting Average") +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        panel.grid.major.x = element_blank(),
        panel.border = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))
ba_over_exp_graph
ops_over_exp_graph <- ggplot(xg_expected_metrics) +
  geom_segment(aes(x=Batter, xend=Batter, y=ops, yend=exp_ops), color="grey") +
  geom_point(aes(x=Batter, y=ops), color="grey", size=4) +
  geom_point(aes(x=Batter, y=exp_ops), color="green", size=4) +
  theme_light() +
  labs(title = "Bushnell OPS v Expected",
       subtitle = "Grey = Actual OPS     |    Green = Expected OPS",
       x = "Batter",
       y = "OPS") +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        panel.grid.major.x = element_blank(),
        panel.border = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))
ops_over_exp_graph

### Hit Accuracy
xgboost_hit_accuracy <- xg_merged_probabilities |> 
  mutate(hit = ifelse(total_bases > 0, 1, 0),
         ba_error = hit_prob - hit) |> 
  summarise(avg_error = mean(ba_error),
            abs_avg_error = mean(abs(ba_error)))





################ Statcast Accuracy ################
### Checking against MLB's model
### Getting limited at 25,000 results. Need to loop through all players
# batter_ids <- mlb_plate_aps |> 
#   distinct(matchup.batter.id) |> filter(matchup.batter.id!=1)
# statcast_data <- data.frame()
# for(i in nrow(batter_ids):1){
#   print(paste0("Player Num: ", i))
#   temp_statcast <- statcast_search(start_date = "2025-09-01", 
#                                    end_date = "2025-09-28",
#                                    playerid = batter_ids$matchup.batter.id[i], 
#                                    player_type = 'batter') |> 
#     select(-game_date)
#   statcast_data <- rbind(statcast_data, temp_statcast, fill=TRUE)
# }
# write_csv(statcast_data, "model_data/statcast.csv")
statcast_data <- read_csv("model_data/statcast.csv") |> 
  filter(batter!=1)

### Determine Statcast's accuracy for batting average
statcast_batted_balls <- statcast_data |> 
  filter(!is.na(estimated_ba_using_speedangle) & !is.na(events) &
           !events%in%c('strikeout','hit_by_pitch','walk','strikeout_double_play', 'truncated_pa','catcher_interf')) |> 
  mutate(total_bases = case_when(
    events == 'single' ~ 1,
    events == 'double' ~ 2,
    events == 'triple' ~ 3,
    events == 'home_run' ~ 4,
    TRUE ~ 0),
    hit = ifelse(events%in%c('single','double','triple','home_run'),1,0)) |> 
  select(game_pk, at_bat_number, batter, player_name, launch_speed, launch_angle, total_bases, hit, estimated_ba_using_speedangle)
statcast_avg_error <- statcast_batted_balls |> 
  mutate(statcast_ba_error = estimated_ba_using_speedangle-hit) |> 
  summarise(avg_error = mean(statcast_ba_error),
            abs_avg_error = mean(abs(statcast_ba_error)))








### OLD
# ###################### Grandient Boosting Classifier Model ######################
# ### Load Model
# gbm_model <- readRDS("model_data/gbm_model.Rds")
# best_iter <- gbm.perf(gbm_model, method = "cv")
# 
# ### Rename columns in dataset
# bushnell_gbm_data <- bushnell_data |> 
#   rename('hitData.launchSpeed'='ExitSpeed', 'hitData.launchAngle'='Angle')
# 
# ### Run Model & reshape results
# pred_probs <- predict(gbm_model, newdata = bushnell_gbm_data, n.trees = best_iter, type = "response")
# pred_probs <- matrix(pred_probs, nrow = nrow(pred_probs))
# colnames(pred_probs) <- c('prob_0','prob_1','prob_2','prob_3','prob_4')
# 
# ### Model Evaluation
# merged_probabilities <- cbind(join_bushnell_data, pred_probs) |> 
#   mutate(exp_bases = prob_1*1 + prob_2*2 + prob_3*3 + prob_4*4,
#          hit_prob = prob_1 + prob_2 + prob_3 + prob_4,
#          play_type = 'InPlay')