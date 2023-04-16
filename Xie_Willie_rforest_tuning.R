# Random Forest tuning ----

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
rforest_model = rand_forest(
  min_n = tune(),
  mtry = tune()) %>%
  set_engine('ranger') %>%
  set_mode('classification')

# workflow ----
rforest_wflow = workflow() %>%
  add_model(rforest_model) %>%
  add_recipe(basic_recipe)

# set-up tuning grid ----
rforest_params = rforest_wflow %>%
  extract_parameter_set_dials() %>%
  update(mtry = mtry(c(1,20)))

# define tuning grid
rforest_grid = grid_regular(rforest_params, levels = 5)

# Tuning/fitting ----
tic.clearlog()
tic("Random Forest")

rforest_tuned = tune_grid(
  rforest_wflow,
  resamples = data_folds,
  grid = rforest_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = TRUE,
                         parallel_over = "everything"),
  metrics = metric_set(roc_auc)
)

# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
rforest_log <- tic.log(format = FALSE)

rforest_tictoc <- tibble(
  model = rforest_log[[1]]$msg,
  runtime = rforest_log[[1]]$toc - rforest_log[[1]]$tic
)

stopCluster(cores.cluster)

# Write out results & workflow
# write_rds(rforest_tuned, file = 'model_info/rforest/rforest_tuned.rds')
# write_rds(rforest_tictoc, file = 'model_info/rforest/rforest_time.rds')




