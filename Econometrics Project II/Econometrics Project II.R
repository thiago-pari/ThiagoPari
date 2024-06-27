# Required libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(tidyverse)
library(foreign)
library(stargazer)
library(haven)
set.seed(10)

# QUESTION 1:

# Loading dataset
mroz_data <- read_dta("MROZ.dta")

# Prepare the data for the models
mroz_data <- mroz_data %>%
  mutate(
    exper_sq = exper^2,
    inlf = as.factor(inlf)
  )

# Linear regression
linear_model <- lm(inlf ~ nwifeinc + educ + exper + exper_sq + age + kidslt6 + kidsge6, data = mroz_data)

# Probit model
probit_model <- glm(inlf ~ nwifeinc + educ + exper + exper_sq + age + kidslt6 + kidsge6, family = binomial(link = "probit"), data = mroz_data)

# Logit model
logit_model <- glm(inlf ~ nwifeinc + educ + exper + exper_sq + age + kidslt6 + kidsge6, family = binomial(link = "logit"), data = mroz_data)

# Display results using stargazer
stargazer(linear_model, probit_model, logit_model, type = "text", title = "Labor Force Participation Models", out = "models_summary.txt")

# Specific values provided for prediction
new_data <- data.frame(
  nwifeinc = 1,
  educ = 10,
  exper = 5,
  exper_sq = 5^2,  # Square of the experience
  age = 30,
  kidslt6 = 1,
  kidsge6 = 0
)

# Predict probabilities using probit and logit models
probit_prediction <- predict(probit_model, newdata = new_data, type = "response")
logit_prediction <- predict(logit_model, newdata = new_data, type = "response")

# Display the predicted probabilities and values
cat("Predicted Probability of Being in Labor Force (Probit): ", probit_prediction, "\n")
cat("Predicted Probability of Being in Labor Force (Logit): ", logit_prediction, "\n")

#-------------------------------------------------------------------------------------------------

#QUESTION 2

# Load the dataset
teaching_data <- read.csv("TeachingRatings.csv")

# a) Run the specified regression
model_a <- lm(course_eval ~ minority + nnenglish + female + age + I(age^2) + intro + onecredit + female * age + female * I(age^2), data = teaching_data)

# a) Output the summary of the regression
summary_a <- summary(model_a)
print(summary_a)

# b) Calculate the estimated marginal effects for a 35-year-old male and female instructor
beta_age <- coef(summary_a)["age", "Estimate"]
beta_age2 <- coef(summary_a)["I(age^2)", "Estimate"]
beta_female_age <- coef(summary_a)["female:age", "Estimate"]
beta_female_age2 <- coef(summary_a)["female:I(age^2)", "Estimate"]

# b) For a 35-year-old male
marginal_effect_male_35 <- beta_age + 2 * beta_age2 * 35
print(paste("Estimated marginal effect of becoming one year older for a 35-year-old male:", marginal_effect_male_35))

# b) For a 35-year-old female
marginal_effect_female_35 <- beta_age + beta_female_age + (2 * beta_age2 + 2 * beta_female_age2) * 35
print(paste("Estimated marginal effect of becoming one year older for a 35-year-old female:", marginal_effect_female_35))

# c) Run the regression without the coefficients associated with female
model_c <- lm(course_eval ~ minority + nnenglish + age + I(age^2) + intro + onecredit, data = teaching_data)
summary_c <- summary(model_c)
print(summary_c)

# d) Test for significance of female-related coefficients
anova_test <- anova(model_a, model_c)
print(anova_test)

# e) Test the overall significance of the model (a)
anova_test_a <- anova(model_a)
print(anova_test_a)

# e) The output from stargazer to compare models easily
stargazer(model_a, model_c, type = "text")

print("Yes, the regression model (a) does explain course evaluations, as it is statistically significant overall, and several individual predictors contribute significantly to the explanation of course evaluation scores. However, additional factors not included in this model might also influence course evaluations.")
