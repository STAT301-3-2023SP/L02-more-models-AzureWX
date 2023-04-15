# Elastic net tuning ----

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
load('Xie_Willie_L02.R')

# Define model ----
enet_model = logistic_reg(
  penalty = tune(),
  mixture = tune()
  ) %>%
  set_engine('glmnet') %>%
  set_mode('classification')

# workflow ----
enet_wflow = workflow() %>%
  add_model(enet_model) %>%
  add_recipe(interact_recipe)

# set-up tuning grid ----
enet_params = enet_wflow %>%
  extract_parameter_set_dials()

# define tuning grid
enet_grid = grid_regular(enet_params, levels = 5)

# Tuning/fitting ----
tic.clearlog()
tic("Elastic Net")

enet_tuned = tune_grid(
  enet_wflow,
  resamples = data_folds,
  grid = enet_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = TRUE,
                         parallel_over = "everything"),
  metrics = metric_set(roc_auc)
)

# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
enet_log <- tic.log(format = FALSE)

enet_tictoc <- tibble(
  model = enet_log[[1]]$msg,
  runtime = enet_log[[1]]$toc - enet_log[[1]]$tic
)

# Write out results & workflow
# write_rds(enet_tuned, file = 'model_info/enet/enet_tuned.rds')
# write_rds(enet_tictoc, file = 'model_info/enet/enet_time.rds')