### EDA
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

# modeling

library(tidymodels)

ames <- 
  make_ames() %>% 
  dplyr::select(-matches("Qu"))

set.seed(4595)
data_split <- initial_split(ames, strata = "Sale_Price")
ames_train <- training(data_split)
ames_test  <- testing(data_split)

lm_mod <- 
  linear_reg() %>% 
  set_engine("lm")

perf_metrics <- metric_set(rmse, rsq, ccc)

# feature engineering

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

# customize neighborhood grouping
# step_mutate

mod_rec <- recipe(
  Sale_Price ~ Longitude + Latitude + Neighborhood, 
  data = ames_train
) %>%
  step_log(Sale_Price, base = 10) %>%
  step_dummy(all_nominal()) %>%
  step_nzv(all_numeric())

mod_rec_trained <- prep(mod_rec, training = ames_train, verbose = TRUE)
ames_test_dummies <- bake(mod_rec_trained, new_data = ames_test)
names(ames_test_dummies)
