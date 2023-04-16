# Boosted tree tuning ----

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
btree_model = boost_tree(
  min_n = tune(),
  mtry = tune(),
  learn_rate = tune()) %>%
  set_engine('xgboost') %>%
  set_mode('classification')

# workflow ----
btree_wflow = workflow() %>%
  add_model(btree_model) %>%
  add_recipe(basic_recipe)

# set-up tuning grid ----
btree_params = btree_wflow %>%
  extract_parameter_set_dials() %>%
  update(mtry = mtry(c(1,20)))

# define tuning grid
btree_grid = grid_regular(btree_params, levels = 5)

# Tuning/fitting ----
tic.clearlog()
tic("Boosted Tree")

btree_tuned = tune_grid(
  btree_wflow,
  resamples = data_folds,
  grid = btree_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = TRUE,
                         parallel_over = "everything"),
  metrics = metric_set(roc_auc)
)

# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
btree_log <- tic.log(format = FALSE)

btree_tictoc <- tibble(
  model = btree_log[[1]]$msg,
  runtime = btree_log[[1]]$toc - btree_log[[1]]$tic
)

stopCluster(cores.cluster)

# Write out results & workflow
write_rds(btree_tuned, file = 'model_info/btree/btree_tuned.rds')
write_rds(btree_tictoc, file = 'model_info/btree/btree_time.rds')




