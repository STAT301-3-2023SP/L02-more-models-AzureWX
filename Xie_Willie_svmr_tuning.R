# SVM polynomial tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)
library(doParallel)

# handle common conflicts
tidymodels_prefer()

# detectCores(logical = FALSE)
cores.cluster = makePSOCKcluster(4)
registerDoParallel(cores.cluster)

# load required objects ----
source('Xie_Willie_L02.R')

# Define model ----
svmr_model = svm_rbf(
  cost = tune(),
  rbf_sigma = tune()) %>%
  set_engine('kernlab') %>%
  set_mode('classification')

# workflow ----
svmr_wflow = workflow() %>%
  add_model(svmr_model) %>%
  add_recipe(basic_recipe)

# set-up tuning grid ----
svmr_params = svmr_wflow %>%
  extract_parameter_set_dials()

# define tuning grid
svmr_grid = grid_regular(svmr_params, levels = 5)

# Tuning/fitting ----
tic.clearlog()
tic("SVM (Radial)")

svmr_tuned = tune_grid(
  svmr_wflow,
  resamples = data_folds,
  grid = svmr_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = TRUE,
                         parallel_over = "everything"),
  metrics = metric_set(roc_auc)
)

# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
svmr_log <- tic.log(format = FALSE)

svmr_tictoc <- tibble(
  model = svmr_log[[1]]$msg,
  runtime = svmr_log[[1]]$toc - svmr_log[[1]]$tic
)

stopCluster(cores.cluster)

# Write out results & workflow
# write_rds(svmr_tuned, file = 'model_info/svmr/svmr_tuned.rds')
# write_rds(svmr_tictoc, file = 'model_info/svmr/svmr_time.rds')




