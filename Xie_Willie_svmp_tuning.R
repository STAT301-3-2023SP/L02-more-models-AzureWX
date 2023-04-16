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
svmp_model = svm_poly(
  cost = tune(),
  degree = tune(),
  scale_factor = tune()) %>%
  set_engine('kernlab') %>%
  set_mode('classification')

# workflow ----
svmp_wflow = workflow() %>%
  add_model(svmp_model) %>%
  add_recipe(basic_recipe)

# set-up tuning grid ----
svmp_params = svmp_wflow %>%
  extract_parameter_set_dials()

# define tuning grid
svmp_grid = grid_regular(svmp_params, levels = 5)

# Tuning/fitting ----
tic.clearlog()
tic("SVM (Polynomial)")

svmp_tuned = tune_grid(
  svmp_wflow,
  resamples = data_folds,
  grid = svmp_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = TRUE,
                         parallel_over = "everything"),
  metrics = metric_set(roc_auc)
)

# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
svmp_log <- tic.log(format = FALSE)

svmp_tictoc <- tibble(
  model = svmp_log[[1]]$msg,
  runtime = svmp_log[[1]]$toc - svmp_log[[1]]$tic
)

stopCluster(cores.cluster)

# Write out results & workflow
# write_rds(svmp_tuned, file = 'model_info/svmp/svmp_tuned.rds')
# write_rds(svmp_tictoc, file = 'model_info/svmp/svmp_time.rds')




