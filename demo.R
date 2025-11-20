# --- Packages ---in
library(ResourceSelection)
library(raster)
library(dplyr)
library(car)
library(sandwich)
library(ggplot2)
library(readr)
library(pROC)
library(caret)
library(randomForest)
library(mgcv)
library(xgboost)
library(Matrix)
library(lmtest) 
library(gridExtra)
library(vip)
map <- read_csv("C:/Users/nph1u24/OneDrive - University of Southampton/MSc Bussiness Analytics and Finance/Project/Data/UK House price/RProject/RProject/map.csv", 
                col_types = cols(Flooded = col_factor(levels = c("0", 
                                                                  "1")), LULC = col_factor(levels = c("1", 
                                                                                                      "2", "3", "4", "5", "6", "7", "8", 
                                                                                                      "9", "10", "11", "12", "13", "14", 
                                                                                                      "15", "16", "17", "18", "19", "20", 
                                                                                                      "21")), Soil = col_skip(), Soil_reclass = col_factor(levels = c("A", 
                                                                                                                                                                      "B", "C", "D", "E", "F", "G"))))
df <- map
summary(df)





# --- Stratified train/test split (preserve class ratio) ---
set.seed(36544086)
train_ratio <- 0.70
idx_0 <- which(df$Flooded == "0")
idx_1 <- which(df$Flooded == "1")

train_0 <- sample(idx_0, size = floor(train_ratio * length(idx_0)))
train_1 <- sample(idx_1, size = floor(train_ratio * length(idx_1)))

train_idx <- c(train_0, train_1)
train_df  <- df[train_idx, ]
test_df   <- df[-train_idx, ]

# --- Formula (same predictors) ---
form <- as.formula(
  Flooded ~ Curvature + Flow + HAND + Rainfall +
    Elevation + Slope + TWI + WaterDist +
    Soil_reclass + LULC
)

# --- Balance the training set (downsample majority class) ---
pos <- dplyr::filter(train_df, Flooded == "1")
neg <- dplyr::filter(train_df, Flooded == "0") %>% dplyr::sample_n(nrow(pos))
train_bal <- dplyr::bind_rows(pos, neg)

# Ensure factor levels are consistent
train_bal$Soil_reclass <- factor(train_bal$Soil_reclass, levels = levels(df$Soil_reclass))
train_bal$LULC         <- factor(train_bal$LULC,         levels = levels(df$LULC))


# 1) LOGISTIC REGRESSION (GLM)
# ===============================
log_model <- glm(form, data = train_bal, family = binomial)

# Predict probabilities on untouched test set
test_df$prob_logit <- predict(log_model, newdata = test_df, type = "response")

roc_logit <- roc(response = test_df$Flooded,
                  predictor = test_df$prob_logit,
                  levels = c("0","1"))
auc_logit <- auc(roc_logit)

# Optimal threshold (Youden)
thr_logit <- coords(roc_logit, "best", best.method = "youden")$threshold
test_df$pred_logit <- ifelse(test_df$prob_logit > thr_logit, "1", "0")

cm_logit <- confusionMatrix(
  factor(test_df$pred_logit, levels = c("0","1")),
  factor(test_df$Flooded,    levels = c("0","1")),
  positive = "1"
)



summary(log_model)
# Odds ratios and 95% CI
exp(cbind(OR = coef(log_model), confint(log_model)))
#Pseudo R2
library(pscl)
pR2(log_model)
#Goodess of fit
library(ResourceSelection)
hoslem.test(log_model$y, fitted(log_model), g = 10)



# a) Multicollinearity
vif(log_model)

#b)heteroskedasticity
bptest(log_model)

rob_vcov <- vcovHC(log_model, type = "HC3")  # HC0/HC1/HC2/HC3 available
robust_table <- coeftest(log_model, vcov. = rob_vcov)
print(robust_table)





# ===============================
# 2) RANDOM FOREST (RF)
# ===============================
set.seed(36544086)
rf_model <- randomForest(
  form,
  data = train_bal,
  ntree = 500,
  mtry = 3,
  importance = TRUE
)

# Predict probabilities on test set
# type="prob" returns a two-column matrix (cols named "0" and "1")
rf_prob <- predict(rf_model, newdata = test_df, type = "prob")
test_df$prob_rf <- rf_prob[, "1"]

roc_rf  <- roc(response = test_df$Flooded,
                predictor = test_df$prob_rf,
                levels = c("0","1"))
auc_rf  <- auc(roc_rf)

thr_rf <- coords(roc_rf, "best", best.method = "youden")$threshold
test_df$pred_rf <- ifelse(test_df$prob_rf > thr_rf, "1", "0")
thr_rf
cm_rf <- confusionMatrix(
  factor(test_df$pred_rf, levels = c("0","1")),
  factor(test_df$Flooded, levels = c("0","1")),
  positive = "1"
)

thr_logit
thr_rf

vip(rf_model)

rf_model  # prints OOB error and confusion matrix
#Variable Importance
varImpPlot(rf_model)

# Extract importance (MeanDecreaseGini)
var_imp <- importance(rf_model)[, "MeanDecreaseGini"]

# Convert to dataframe
imp_df <- data.frame(
  Variable = names(var_imp),
  Importance = var_imp
)
# Bar chart with ggplot2
ggplot(imp_df, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +   # flip for horizontal bars (easier to read)
  labs(
    title = "Variable Importance (Random Forest)",
    x = "Variable",
    y = "Importance"
  ) +
  theme_minimal()

# 3) REPORT + PLOTS
# ===============================
cat("\n=== AUCs ===\n")
cat(sprintf("Logistic AUC: %.4f | Threshold: %.4f\n", as.numeric(auc_logit), thr_logit))
cat(sprintf("RF       AUC: %.4f | Threshold: %.4f\n", as.numeric(auc_rf),    thr_rf))

cat("\n=== Confusion Matrix: Logistic ===\n")
print(cm_logit)

cat("\n=== Confusion Matrix: Random Forest ===\n")
print(cm_rf)

# ROC plot: both models
plot(roc_logit, col = "blue", lwd = 2,
      main = "ROC Curves: Logistic vs Random Forest")
plot(roc_rf,   col = "darkgreen", lwd = 2, add = TRUE)
legend("bottomright",
        legend = c(paste("Logit AUC =", round(auc_logit, 3)),
                  paste("RF AUC   =", round(auc_rf, 3))),
        col = c("blue","darkgreen"), lwd = 2, bty = "n")


# --- Compute F1 scores ---
f1_logit <- F_meas(data = factor(test_df$pred_logit, levels = c("0","1")),
                    reference = factor(test_df$Flooded, levels = c("0","1")),
                    relevant = "1")

f1_rf <- F_meas(data = factor(test_df$pred_rf, levels = c("0","1")),
                reference = factor(test_df$Flooded, levels = c("0","1")),
                relevant = "1")

# --- Simplified comparison table ---
comparison <- data.frame(
  Metric = c("Accuracy", "Balanced Accuracy", "Sensitivity (Recall)", 
              "Specificity", "Precision (PPV)", "F1 Score", "AUC"),
  Logistic_Regression = c(cm_logit$overall["Accuracy"],
                          cm_logit$byClass["Balanced Accuracy"],
                          cm_logit$byClass["Sensitivity"],
                          cm_logit$byClass["Specificity"],
                          cm_logit$byClass["Pos Pred Value"],
                          f1_logit,
                          as.numeric(auc_logit)),
  Random_Forest = c(cm_rf$overall["Accuracy"],
                    cm_rf$byClass["Balanced Accuracy"],
                    cm_rf$byClass["Sensitivity"],
                    cm_rf$byClass["Specificity"],
                    cm_rf$byClass["Pos Pred Value"],
                    f1_rf,
                    as.numeric(auc_rf))
)

print(comparison)








