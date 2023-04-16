# Neural network tuning ----

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
nn_model = mlp(
  hidden_units = tune(),
  penalty = tune()) %>%
  set_engine('nnet') %>%
  set_mode('classification')

# workflow ----
nn_wflow = workflow() %>%
  add_model(nn_model) %>%
  add_recipe(basic_recipe)

# set-up tuning grid ----
nn_params = nn_wflow %>%
  extract_parameter_set_dials()

# define tuning grid
nn_grid = grid_regular(nn_params, levels = 5)

# Tuning/fitting ----
tic.clearlog()
tic("Neural Network")

nn_tuned = tune_grid(
  nn_wflow,
  resamples = data_folds,
  grid = nn_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = TRUE,
                         parallel_over = "everything"),
  metrics = metric_set(roc_auc)
)

# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
nn_log <- tic.log(format = FALSE)

nn_tictoc <- tibble(
  model = nn_log[[1]]$msg,
  runtime = nn_log[[1]]$toc - nn_log[[1]]$tic
)

stopCluster(cores.cluster)

# Write out results & workflow
# write_rds(nn_tuned, file = 'model_info/nn/nn_tuned.rds')
# write_rds(nn_tictoc, file = 'model_info/nn/nn_time.rds')




