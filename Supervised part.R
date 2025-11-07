library(data.table)

#Loading Data
supervised <- read.csv("F:/University/Projects/Data Science/Statistical Learning/Maven Project/supervised.csv")

###Data preparation

str(supervised)

# Convert 'became_member_on' to Date type
supervised$became_member_on <- as.Date(supervised$became_member_on)

# Convert 'gender' into a factor first (good practice for categorical variables)
supervised$gender <- factor(supervised$gender, levels = c("M", "F", "O"))

# Use model.matrix to create one-hot encoded variables
# model.matrix automatically converts factors into dummy/one-hot encoded variables
one_hot_gender <- model.matrix(~ gender - 1, data = supervised)

# Append one-hot encoded columns back to the original dataset
supervised <- cbind(supervised, one_hot_gender)

# Drop the original 'gender' column
supervised$gender <- NULL


# Creating a feature for Membership Duration (in days)
supervised$membership_duration <- as.numeric(as.Date("2024-09-15") - supervised$became_member_on)

# Bining membership_duration into 3 categories: "short-term," "medium-term," and "long-term"
# We'll use quantiles to divide the data into roughly equal groups
supervised$membership_category <- cut(supervised$membership_duration,
                                      breaks = quantile(supervised$membership_duration, probs = seq(0, 1, by = 1/3), na.rm = TRUE),
                                      labels = c("short-term", "medium-term", "long-term"),
                                      include.lowest = TRUE)

# Z-score Standardization (mean = 0, sd = 1)
supervised$membership_duration_zscore <- (supervised$membership_duration - mean(supervised$membership_duration, na.rm = TRUE)) /
  sd(supervised$membership_duration, na.rm = TRUE)

# Drop the original 'became_member_on' column
supervised$became_member_on <- NULL

# Binning age into categories: "young adult," "middle-aged," "senior"
supervised$age_group <- cut(supervised$age,
                            breaks = c(-Inf, 25, 45, 65, Inf),
                            labels = c("young adult", "middle-aged", "senior", "elder"),
                            right = FALSE)

# Binning income into categories: "low," "medium," and "high"
supervised$income_group <- cut(supervised$income,
                               breaks = quantile(supervised$income, probs = seq(0, 1, by = 1/3), na.rm = TRUE),
                               labels = c("low", "medium", "high"),
                               include.lowest = TRUE)

# Drop the original 'age' column
supervised$age <- NULL

# Drop the original 'income' column
supervised$income <- NULL

# Ordinal encoding for 'age_group'
supervised$age_group <- factor(supervised$age_group, ordered = TRUE,
                               levels = c("young adult", "middle-aged", "senior", "elder"))

# Ordinal encoding for 'income_group'
supervised$income_group <- factor(supervised$income_group, ordered = TRUE,
                                  levels = c("low", "medium", "high"))

str(supervised)


# Create a target variable based on which offer type the customer completed the most
supervised$preferred_offer_type <- apply(supervised[, c("offer_type_bogo_completed",
                                                        "offer_type_discount_completed",
                                                        "num_completions_after_info")],
                                         1, function(x) {
                                           # Identify which offer type has the maximum completion
                                           offer_types <- c("bogo", "discount", "informational")
                                           max_index <- which.max(x)
                                           offer_types[max_index]
                                         })

supervised[is.na(supervised)] <- 0


# Check the distribution of the target variable
table(supervised$preferred_offer_type)

# Convert preferred_offer_type to a factor
supervised$preferred_offer_type <- as.factor(supervised$preferred_offer_type)


cols_to_drop <- c("num_offers_received",
                  "num_offers_viewed",
                  "num_offers_completed",
                  "offer_type_bogo_completed",
                  "offer_type_discount_completed",
                  "num_completions_after_info")

supervised_noleak <- supervised[, setdiff(names(supervised), cols_to_drop)]




###MODEL
# Install the necessary packages
install.packages("xgboost")
install.packages("caret")
install.packages("Matrix")
install.packages("dplyr")

# Load the libraries
library(xgboost)
library(caret)
library(Matrix)
library(dplyr)
library(mlr3)

# ===== Packages =====
pkgs_needed <- c("mlr3verse","mlr3learners","mlr3tuning","mlr3pipelines","paradox","nnet")
newp <- pkgs_needed[!(pkgs_needed %in% installed.packages()[, "Package"])]
if (length(newp)) install.packages(newp, dependencies = TRUE)
invisible(lapply(pkgs_needed, library, character.only = TRUE))

set.seed(42)

# ===== 0) Task setup =====
# Ensure target is factor
supervised_noleak$preferred_offer_type <- as.factor(supervised$preferred_offer_type)
# Drop IDs if present
if ("customer_id" %in% names(supervised_noleak)) supervised_noleak$customer_id <- NULL

task_all <- TaskClassif$new(id = "offers_all", backend = supervised_noleak, target = "preferred_offer_type")

# ===== 1) 80/20 external split =====
# stratified split to keep class ratios
str(supervised)

dt <- as.data.table(supervised_noleak)
dt[, row_id := .I]

dt <- dt[!is.na(preferred_offer_type)]
dt[, preferred_offer_type := droplevels(preferred_offer_type)]

# 80% از هر کلاس با استفاده از ایندکس‌های ردیفی (.I)
train_ids <- dt[, sample(.I, max(1, ceiling(.N * 0.8))), by = preferred_offer_type]$V1
train_ids <- sort(unique(train_ids))
test_ids  <- setdiff(seq_len(nrow(dt)), train_ids)


task_train <- TaskClassif$new("offers_train", backend = dt[train_ids], target = "preferred_offer_type")
task_test  <- TaskClassif$new("offers_test",  backend = dt[test_ids],  target = "preferred_offer_type")


# ===== 2) Build preprocessing pipeline (no leakage) =====
po_imp_num <- po("imputemean")
po_imp_cat <- po("imputemode")
po_fix     <- po("fixfactors")
po_encode  <- po("encode")
po_const   <- po("removeconstants")
po_scale   <- po("scale")

# ===== 3) Candidate learners + search spaces (inner loop) =====
# XGBoost
lrn_xgb <- lrn("classif.xgboost", eval_metric = "mlogloss", booster = "gbtree");lrn_xgb$id <- "xgb"
g_xgb <- as_learner(po_imp_num %>>% po_imp_cat %>>% po_fix %>>% po_encode %>>% po_const %>>% po_scale %>>% lrn_xgb)

space_xgb <- ps(
  xgb.nrounds          = p_int(100, 500),
  xgb.eta              = p_dbl(0.01, 0.2),
  xgb.max_depth        = p_int(2, 8),
  xgb.subsample        = p_dbl(0.6, 1.0),
  xgb.colsample_bytree = p_dbl(0.6, 1.0),
  xgb.min_child_weight = p_dbl(1, 10),
  xgb.gamma            = p_dbl(0, 5)
)

# Random Forest (ranger)
lrn_rf <- lrn("classif.ranger");lrn_rf$id  <- "rf"
g_rf  <- as_learner(po_imp_num %>>% po_imp_cat %>>% po_fix %>>% po_encode %>>% po_const %>>% po_scale %>>% lrn_rf)

space_rf <- ps(
  rf.num.trees       = p_int(200, 800),
  rf.mtry            = p_int(1, max(1, ncol(supervised)-1)),
  rf.min.node.size   = p_int(1, 20),
  rf.sample.fraction = p_dbl(0.5, 1.0)
)

# Logistic regression multinom
lrn_mn <- lrn("classif.multinom");lrn_mn$id  <- "mn"
g_mn  <- as_learner(po_imp_num %>>% po_imp_cat %>>% po_fix %>>% po_encode %>>% po_const %>>% po_scale %>>% lrn_mn)

space_mn <- ps(
  mn.decay = p_dbl(0, 0.1),
  mn.maxit = p_int(100, 500)
)

# SVM (radial)
lrn_svm <- lrn(
  "classif.svm",
  type   = "C-classification",
  kernel = "radial"
)
lrn_svm$id <- "svm"

# گراف + تبدیل به learner
g_svm <- as_learner(
  po_imp_num %>>% po_imp_cat %>>% po_fix %>>% po_encode %>>% po_const %>>% po_scale %>>% lrn_svm
)
g_svm$predict_type <- "prob"
space_svm <- ps(
  svm.cost  = p_dbl(0.1, 10),
  svm.gamma = p_dbl(1e-3, 1)
)

# ===== 4) Inner/Outer resampling & AutoTuner helper =====
inner_rs <- rsmp("cv", folds = 3); if ("stratify" %in% inner_rs$param_set$ids()) inner_rs$param_set$values$stratify <- TRUE
outer_rs <- rsmp("cv", folds = 5); if ("stratify" %in% outer_rs$param_set$ids()) outer_rs$param_set$values$stratify <- TRUE
terminator <- trm("evals", n_evals = 10)
tuner      <- tnr("random_search")


make_at <- function(glrn, space, id) {
  AutoTuner$new(
    learner = glrn,
    resampling = inner_rs,
    measure = msr("classif.acc"),
    search_space = space,
    terminator = terminator,
    tuner = tuner,
    id = id,
    store_tuning_instance = TRUE,
    store_benchmark_result = TRUE
  )
}

at_xgb <- make_at(g_xgb, space_xgb, "xgb")
at_rf  <- make_at(g_rf,  space_rf,  "ranger")
at_svm <- make_at(g_svm, space_svm, "svm")
at_mn  <- make_at(g_mn,  space_mn,  "multinom")

# ===== 5) Nested CV on 80% train to find winner =====
design <- benchmark_grid(
  tasks = task_train,
  learners = list(at_xgb, at_rf, at_mn, at_svm),
  resamplings = outer_rs
)
bmr <- benchmark(design, store_models = TRUE)



# Aggregate unbiased outer-CV metrics on train split
agg <- as.data.table(bmr$aggregate(list(
  msr("classif.acc"),
  msr("classif.bacc"),
  msr("classif.fbeta")  # F1 macro (beta=1)
)))
print(agg[order(-classif.acc)])

# Pick winner by outer-CV accuracy
perf_by_learner <- agg[, .(acc = mean(classif.acc)), by = learner_id][order(-acc)]
winner_id <- perf_by_learner$learner_id[1]
cat(sprintf("\nSelected winner on 80%%-train (nested CV): %s\n", winner_id))

winner <- switch(winner_id,
                 "xgb"    = at_xgb,
                 "ranger" = at_rf,
                 "glmnet" = at_glm,
                 "svm"    = at_svm
)

# ===== 6) Fit winner on ALL 80% train, then evaluate ONCE on 20% external test =====
set.seed(999)
winner$train(task_train)

pred_test <- winner$predict(task_test)



# Metrics: Accuracy, Precision/Recall/F1 (macro)
acc   <- pred_test$score(msr("classif.acc"))




pd <- as.data.table(pred_test)
classes <- levels(pd$truth)

# Confusion matrix using caret
conf_mat <- confusionMatrix(pd$response, pd$truth)

# Accuracy
accuracy <- conf_mat$overall['Accuracy']

# Precision, Recall, F1 for each class
precision <- conf_mat$byClass[,'Precision']
recall <- conf_mat$byClass[,'Recall']
f1 <- conf_mat$byClass[,'F1']

# Print results
cat("Accuracy:", accuracy, "\n\n")
cat("Precision:\n"); print(precision)
cat("\nRecall:\n"); print(recall)
cat("\nF1 Score:\n"); print(f1)

macro_precision <- mean(precision, na.rm = TRUE)
macro_recall <- mean(recall, na.rm = TRUE)
macro_f1 <- mean(f1, na.rm = TRUE)

cat("\nMacro Precision:", macro_precision)
cat("\nMacro Recall:", macro_recall)
cat("\nMacro F1:", macro_f1, "\n")










# Confusion matrix (Test)
cm <- with(pred_test$data, table(truth, response))
print(cm)





# ===== 8) Save artifacts =====
saveRDS(winner, "winner_autotuned_on_80pct.rds")
saveRDS(pred_test, "external_test_predictions.rds")
fwrite(as.data.table(cm), "confusion_matrix_test.csv")
