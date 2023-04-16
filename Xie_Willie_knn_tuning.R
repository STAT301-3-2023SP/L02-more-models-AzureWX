# K-nearest neighbors tuning ----

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
knn_model = nearest_neighbor(
  neighbors = tune()) %>%
  set_engine('kknn') %>%
  set_mode('classification')

# workflow ----
knn_wflow = workflow() %>%
  add_model(knn_model) %>%
  add_recipe(basic_recipe)

# set-up tuning grid ----
knn_params = knn_wflow %>%
  extract_parameter_set_dials()

# define tuning grid
knn_grid = grid_regular(knn_params, levels = 5)

# Tuning/fitting ----
tic.clearlog()
tic("K nearest")

knn_tuned = tune_grid(
  knn_wflow,
  resamples = data_folds,
  grid = knn_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = TRUE,
                         parallel_over = "everything"),
  metrics = metric_set(roc_auc)
)

# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
knn_log <- tic.log(format = FALSE)

knn_tictoc <- tibble(
  model = knn_log[[1]]$msg,
  runtime = knn_log[[1]]$toc - knn_log[[1]]$tic
)

stopCluster(cores.cluster)

# Write out results & workflow
# write_rds(knn_tuned, file = 'model_info/knn/knn_tuned.rds')
# write_rds(knn_tictoc, file = 'model_info/knn/knn_time.rds')




