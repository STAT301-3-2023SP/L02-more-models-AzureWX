# MARS tuning ----

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
mars_model = mars(
  num_terms = tune(),
  prod_degree = tune()) %>%
  set_engine('earth') %>%
  set_mode('classification')

# workflow ----
mars_wflow = workflow() %>%
  add_model(mars_model) %>%
  add_recipe(basic_recipe)

# set-up tuning grid ----
mars_params = mars_wflow %>%
  extract_parameter_set_dials() %>%
  update(num_terms = num_terms(c(1,15)))

# define tuning grid
mars_grid = grid_regular(mars_params, levels = 5)

# Tuning/fitting ----
tic.clearlog()
tic("MARS")

mars_tuned = tune_grid(
  mars_wflow,
  resamples = data_folds,
  grid = mars_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = TRUE,
                         parallel_over = "everything"),
  metrics = metric_set(roc_auc)
)

# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
mars_log <- tic.log(format = FALSE)

mars_tictoc <- tibble(
  model = mars_log[[1]]$msg,
  runtime = mars_log[[1]]$toc - mars_log[[1]]$tic
)

stopCluster(cores.cluster)

# Write out results & workflow
# write_rds(mars_tuned, file = 'model_info/mars/mars_tuned.rds')
# write_rds(mars_tictoc, file = 'model_info/mars/mars_time.rds')




