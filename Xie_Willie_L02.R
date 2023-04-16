# Load package(s)
library(tidymodels)
library(tidyverse)

# handle common conflicts
tidymodels_prefer()

# Seed
set.seed(3013)

## load data
wildfires_dat <- read_csv("data/wildfires.csv") %>%
  janitor::clean_names() %>%
  mutate(
    winddir = factor(winddir, levels = c("N", "NE", "E", "SE", "S", "SW", "W", "NW")),
    traffic = factor(traffic, levels = c("lo", "med", "hi")),
    wlf = factor(wlf, levels = c(1, 0), labels = c("yes", "no"))
  ) %>%
  select(-burned)

# split data
data_split = initial_split(wildfires_dat, prop = 0.80, strata = wlf)
# training and testing sets
data_train = training(data_split)
data_test = testing(data_split)
# validation sets
data_folds = vfold_cv(data_train, 5, 3)

# recipe
basic_recipe = recipe(wlf ~ ., data = data_train) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors())
# View(basic_recipe %>% prep() %>% bake(new_data = NULL))

interact_recipe = recipe(wlf ~ ., data = data_train) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_interact(~ all_predictors():all_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors())
# View(interact_recipe %>% prep() %>% bake(new_data = NULL))
  

