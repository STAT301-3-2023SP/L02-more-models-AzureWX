# Load package(s)
library(tidymodels)
library(tidyverse)
source('Xie_Willie_L02.R')

# load models
enet_tuned = read_rds('model_info/enet/enet_tuned.rds')
enet_time = read_rds('model_info/enet/enet_time.rds')

knn_tuned = read_rds('model_info/knn/knn_tuned.rds')
knn_time = read_rds('model_info/knn/knn_time.rds')

rforest_tuned = read_rds('model_info/rforest/rforest_tuned.rds')
rforest_time = read_rds('model_info/rforest/rforest_time.rds')

btree_tuned = read_rds('model_info/btree/btree_tuned.rds')
btree_time = read_rds('model_info/btree/btree_time.rds')

svmp_tuned = read_rds('model_info/svmp/svmp_tuned.rds')
svmp_time = read_rds('model_info/svmp/svmp_time.rds')

svmr_tuned = read_rds('model_info/svmr/svmr_tuned.rds')
svmr_time = read_rds('model_info/svmr/svmr_time.rds')

nn_tuned = read_rds('model_info/nn/nn_tuned.rds')
nn_time = read_rds('model_info/nn/nn_time.rds')

mars_tuned = read_rds('model_info/mars/mars_tuned.rds')
mars_time = read_rds('model_info/mars/mars_time.rds')

# compare models
time_sum = bind_rows(
  enet_time, knn_time,
  rforest_time, btree_time,
  svmp_time, svmr_time,
  nn_time, mars_time)

models_sum = bind_rows(
  show_best(enet_tuned, metric = "roc_auc") %>% head(1) %>%
    mutate(model = 'Elastic Net') %>%
    select(model, .metric, mean, std_err, penalty, mixture),
  show_best(knn_tuned, metric = "roc_auc") %>% head(1) %>%
    mutate(model = 'K nearest') %>%
    select(model, .metric, mean, std_err, neighbors),
  show_best(rforest_tuned, metric = "roc_auc") %>% head(1) %>%
    mutate(model = 'Random Forest') %>%
    select(model, .metric, mean, std_err, mtry, min_n),
  show_best(btree_tuned, metric = "roc_auc") %>% head(1) %>%
    mutate(model = 'Boosted Tree') %>%
    select(model, .metric, mean, std_err, mtry, min_n, learn_rate),
  show_best(svmp_tuned, metric = "roc_auc") %>% head(1) %>%
    mutate(model = 'SVM (Polynomial)') %>%
    select(model, .metric, mean, std_err, cost, degree, scale_factor),
  show_best(svmr_tuned, metric = "roc_auc") %>% head(1) %>%
    mutate(model = 'SVM (Radial)') %>%
    select(model, .metric, mean, std_err, cost, rbf_sigma),
  show_best(nn_tuned, metric = "roc_auc") %>% head(1) %>%
    mutate(model = 'Neural Network') %>%
    select(model, .metric, mean, std_err, hidden_units, penalty),
  show_best(mars_tuned, metric = "roc_auc") %>% head(1) %>%
    mutate(model = 'MARS') %>%
    select(model, .metric, mean, std_err, num_terms, prod_degree)
)

complete_sum = models_sum %>%
  inner_join(time_sum, by = 'model') %>%
  relocate(runtime, .after = model)
# write_rds(complete_sum, 'model_info/model_sum.rds')
complete_sum

ggplot(complete_sum, aes(model, mean)) +
  geom_point() +
  geom_errorbar(aes(ymin = mean - 1.96*std_err, ymax = mean + 1.96*std_err)) +
  scale_y_continuous(breaks = seq(0.5, 1, 0.02)) +
  scale_x_discrete(limits = (complete_sum %>% arrange(desc(mean)))$model) +
  coord_flip() +
  labs(
    title = 'Model Performance',
    x = 'Model', y = 'Roc Auc'
  )

# final model
enet_model = logistic_reg(
  penalty = tune(),
  mixture = tune()) %>%
  set_engine('glmnet') %>%
  set_mode('classification')
enet_wflow = workflow() %>%
  add_model(enet_model) %>%
  add_recipe(interact_recipe)
final_wflow = enet_wflow %>%
  finalize_workflow(select_best(enet_tuned, metric = 'roc_auc'))
final_fit = last_fit(
  object = final_wflow,
  split = data_split,
  metrics = metric_set(precision, f_meas, accuracy, roc_auc))
# write_rds(final_fit, 'model_info/final_model.rds')

final_fit %>% collect_metrics()
final_fit %>% collect_predictions() %>%
  conf_mat(truth = wlf, estimate = .pred_class) %>%
  autoplot(type = 'heatmap')
final_fit %>% collect_predictions() %>%
  roc_curve(wlf, .pred_yes) %>%
  autoplot()





