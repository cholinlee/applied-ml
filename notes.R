# EDA ==================
library(AmesHousing)
ames <- make_ames()

ggplot(ames, aes(x = Bedroom_AbvGr, y = Sale_Price, fill = Condition_1)) + 
  geom_col(position = "dodge") +
  facet_wrap(~Neighborhood)

ggplot(ames, aes(x = Full_Bath/Bedroom_AbvGr, y = Sale_Price)) + 
  geom_point(stat = "identity") 

ggplot(ames, aes(x = Gr_Liv_Area, y = Sale_Price )) +
  geom_point(stat = "identity")+
  geom_smooth()+
  # geom_histogram()+
  scale_x_log10() + 
  scale_y_log10() + 
  facet_wrap(~Neighborhood)

ames %>%
  group_by(Neighborhood) %>%
  summarise(Sale_Price = mean(Sale_Price, na.rm = T)) %>%
  arrange(-Sale_Price)

# modeling ============================

library(tidymodels)

ames <- 
  make_ames() %>% 
  # remove subjective quality assessment
  dplyr::select(-matches("Qu"))

set.seed(4595)

# data split -----------------
# split the sample based on outcome (can also do split by predictors, random sampling, etc. )
# can also change proportion, default = 0.75
data_split <- initial_split(ames, strata = "Sale_Price")

ames_train <- training(data_split)
ames_test  <- testing(data_split)


# (model fitting) old fashion ------------
# linear
simple_lm <- lm(log10(Sale_Price) ~ Longitude + Latitude, data = ames_train)
simple_lm_values <- broom::augment(simple_lm)
head(simple_lm_values)

# parsnip ----------------
# standardize interface/parameters of different models/engines
# 1. specify the model
spec_lin_reg <- linear_reg()
# 2. set the engine
lm_mod <- set_engine(spec_lin_reg, "lm")
# 3. fit the model
lm_fit <- fit(
  # model
  lm_mod,
  # formula 
  log10(Sale_Price) ~ Longitude + Latitude,
  # data
  data = ames_train
)

lm_fit

# switch interface for formula:
ames_train_log <- ames_train %>%
  # do transformation first
  mutate(Sale_Price_Log = log10(Sale_Price))

fit_xy(
  lm_mod,
  y = ames_train_log$Sale_Price_Log,
  x = ames_train_log %>% dplyr::select(Latitude, Longitude)
)

# switch engine
spec_stan <- 
  spec_lin_reg %>%
  # Engine specific arguments are passed through here
  set_engine("stan", chains = 4, iter = 1000)

# Otherwise, looks exactly the same!
fit_stan <- fit(
  spec_stan,
  # only allow formula interface
  log10(Sale_Price) ~ Longitude + Latitude,
  data = ames_train
)

coef(fit_stan$fit)

# switch models
fit_knn <- 
  nearest_neighbor(mode = "regression", neighbors = 5) %>%
  set_engine("kknn") %>% 
  fit(log10(Sale_Price) ~ Longitude + Latitude, data = ames_train)

fit_knn

# predict -----------------
# skip the resampling part for now
test_pred <- 
  # lm_fit %>%
  fit_knn %>%
  predict(ames_test) %>%
  bind_cols(ames_test) %>%
  mutate(log_price = log10(Sale_Price))

ggplot(test_pred, aes(x = .pred, y = log_price)) + 
  geom_point(stat = "identity") + 
  geom_smooth()

# measure performance ------------

perf_metrics <- yardstick::metric_set(rmse, rsq, ccc)
# A tidy result back:
test_pred  %>% 
  perf_metrics(truth = log_price, estimate = .pred)

# feature engineering ------------
# receipe: specify the required steps 

mod_rec <- recipe(
  Sale_Price ~ Longitude + Latitude + Neighborhood, 
  data = ames_train
) %>%
  step_log(Sale_Price, base = 10) %>%
  # Lump factor levels that occur in 
  # <= 5% of data as "other"
  step_other(Neighborhood, threshold = 0.05) %>%
  # Create dummy variables for _any_ factor variables
  step_dummy(all_nominal())

# use step_mutate to customize neighborhood grouping

mod_rec <- recipe(
  Sale_Price ~ Longitude + Latitude + Neighborhood, 
  data = ames_train
) %>%
  step_log(Sale_Price, base = 10) %>%
  step_dummy(all_nominal()) %>%
  # non-zero variance
  step_nzv(all_numeric())

mod_rec_trained <- prep(mod_rec, training = ames_train, verbose = TRUE)

# processed version of the training data
juice(mod_rec_trained)
# apply to any other datasets
ames_test_dummies <- bake(mod_rec_trained, new_data = ames_test)
head(ames_test_dummies)

# interactions, use ANOVA to test 
mod1 <- lm(log10(Sale_Price) ~ Year_Built + Central_Air,                          data = ames_train)
mod2 <- lm(log10(Sale_Price) ~ Year_Built + Central_Air + Year_Built:Central_Air, data = ames_train)
anova(mod1, mod2)

recipe(Sale_Price ~ Year_Built + Central_Air, data = ames_train) %>%
  step_log(Sale_Price) %>%
  step_dummy(Central_Air) %>%
  step_interact(~ starts_with("Central_Air"):Year_Built) %>%
  prep(training = ames_train) %>%
  juice() %>%
  # select a few rows with different values
  slice(153:157)

# PCA
# Box-Cox
# put it together
ames_rec <- 
  recipe(Sale_Price ~ Bldg_Type + Neighborhood + Year_Built + 
           Gr_Liv_Area + Full_Bath + Year_Sold + Lot_Area +
           Central_Air + Longitude + Latitude,
         data = ames_train) %>%
  # log transformation
  step_log(Sale_Price, base = 10) %>%
  # boxCox
  step_BoxCox(Lot_Area, Gr_Liv_Area) %>%
  # rare factors
  step_other(Neighborhood, threshold = 0.05)  %>%
  # dummy
  step_dummy(all_nominal()) %>%
  # interactions
  step_interact(~ starts_with("Central_Air"):Year_Built) %>%
  # natural splines
  step_ns(Longitude, Latitude, deg_free = 5)


# workflow ------------------
library(workflows)
ames_wfl <- 
  workflow() %>% 
  add_recipe(ames_rec) %>% 
  add_model(lm_mod)

ames_wfl

ames_wfl_fit <- fit(ames_wfl, ames_train)
predict(ames_wfl_fit, ames_test %>% slice(1:5))

# resampling --------------
# v-fold
set.seed(2453)
cv_splits <- vfold_cv(ames_train) #10-fold is default
cv_splits
cv_splits$splits[[1]] %>% analysis() %>% dim()
cv_splits$splits[[1]] %>% assessment() %>% dim()

knn_mod <- 
  nearest_neighbor(neighbors = 5) %>% 
  set_engine("kknn") %>% 
  set_mode("regression")

# with training set only
knn_wfl <- 
  workflow() %>% 
  add_model(knn_mod) %>% 
  add_formula(log10(Sale_Price) ~ Longitude + Latitude)

fit(knn_wfl, data = ames_train)

# (resampling) old fashion ------------
# with 10 splits
knn_res <-
  cv_splits %>%
  # use purrr::map to use workflow on each splits
  mutate( workflows = map(splits, ~ fit( knn_wfl, data = analysis(.x)) ) ) 

knn_pred <-
  # use map2_dfr to bind result
  map2_dfr(knn_res$workflows, knn_res$splits,
           # use ~ to initiate inline function
           ~ predict(.x, assessment(.y)),
           .id = "fold")

# extract the estimators
prices <-  
  map_dfr(knn_res$splits,  
          ~ assessment(.x) %>% select(Sale_Price)) %>%  
  mutate(Sale_Price = log10(Sale_Price))

rmse_estimates <- 
  knn_pred %>%  
  bind_cols(prices) %>% 
  group_by(fold) %>% 
  do(rmse = rmse(., Sale_Price, .pred)) %>% 
  unnest(cols = c(rmse)) 

mean(rmse_estimates$.estimate)

# tune ===================
library(tune)
easy_eval <- fit_resamples(knn_wfl, resamples = cv_splits, control = control_resamples(save_pred = TRUE))

# get the estimators
collect_predictions(easy_eval) 
collect_metrics(easy_eval)
collect_metrics(easy_eval, summarize = FALSE)

# tuning ----------------
# grid search， dial

penalty()
mixture()

# regular
glmn_param <- parameters(penalty(), mixture())
glmn_grid <- grid_regular(glmn_param, levels = c(10, 5))

# non regular
set.seed(7454)
glmn_sfd <- grid_max_entropy(glmn_param, size = 50)

# The names can be changed:
glmn_set <- parameters(lambda = penalty(), mixture())
# The ranges can also be set by their name:
glmn_set <- 
  update(glmn_set, lambda = penalty(c(-5, -1)))


