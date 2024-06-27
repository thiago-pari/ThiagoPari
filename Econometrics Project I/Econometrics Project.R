library(haven)
data <- read_dta('/Users/thiago.pari/Downloads/morg16 - Copy.dta')

# a) Subset the data
subset_data <- data[data$age >= 25 & data$age <= 65 & data$stfips == 25, c("earnwke", "uhourse", "sex", "age")]

# b) Data cleaning: Remove NA and create a dummy variable
subset_data <- na.omit(subset_data)
subset_data$worked <- ifelse(subset_data$uhourse > 0, 1, 0)

# What is the share of workers working more than 0 hours
share_working <- mean(subset_data$worked)

# Calculate the mean number of hours worked in the restricted sample
mean_hours_worked_restricted <- mean(subset_data$uhourse[subset_data$worked == 1])

# Compare to the overall mean hours worked
mean_hours_worked <- mean(subset_data$uhourse)

# Replacing subset_data to restricted
subset_data_restricted <- subset_data[subset_data$worked == 1, ]

# c) Plot weekly earnings against usual hours
plot(subset_data_restricted$uhourse, subset_data_restricted$earnwke, main = "Weekly Earnings vs. Usual Hours", xlab = "Usual Hours", ylab = "Weekly Earnings")

# d) Create a measure of wages
subset_data_restricted$wages <- subset_data_restricted$earnwke / subset_data_restricted$uhourse

# e) Plot our measure of wages against the usual hours
plot(subset_data_restricted$uhourse, subset_data_restricted$wages, main = "Wages vs. Usual Hours", xlab = "Usual Hours", ylab = "Wages")

# f) Plot for identified male and female
male_data <- subset_data_restricted[subset_data_restricted$sex == "1", ]
female_data <- subset_data_restricted[subset_data_restricted$sex == "2", ]

plot(male_data$uhourse, male_data$wages, col = "gray", main = "Wages vs. Usual Hours by Gender", xlab = "Usual Hours", ylab = "Wages")
points(female_data$uhourse, female_data$wages, col = "red")
lm_male <- lm(wages ~ uhourse, data = male_data)
lm_female <- lm(wages ~ uhourse, data = female_data)

# Adding regression lines
abline(lm_male, col = "blue")
abline(lm_female, col = "magenta")

# Adding a legend to differentiate between the points and lines
legend("topright", legend=c("Male", "Female", "Male Regr. Line", "Female Regr. Line"), col=c("gray", "red", "blue", "magenta"), pch=1, lty=1)

# g) Running regression for male and female
regression_male <- lm(wages ~ age, data = male_data)
regression_female <- lm(wages ~ age, data = female_data)

# h) Reporting results with stargazer
library(stargazer)
stargazer(regression_male, regression_female, type = "text")

# i) # Generating residuals from the wage-age regression for male and female
male_data$residuals_age <- residuals(regression_male)
female_data$residuals_age <- residuals(regression_female)

# Plotting the residuals against hours worked for identified male
plot(male_data$uhourse, male_data$residuals_age, col = "gray", main = "Residuals vs. Usual Hours by Gender", xlab = "Usual Hours", ylab = "Residuals", ylim = c(min(c(male_data$residuals_age, female_data$residuals_age)), max(c(male_data$residuals_age, female_data$residuals_age))))
points(female_data$uhourse, female_data$residuals_age, col = "red")
lm_male_2 <- lm(residuals_age ~ uhourse, data = male_data)
lm_female_2 <- lm(residuals_age ~ uhourse, data = female_data)

# Adding regression lines
abline(lm_male_2, col = "blue")
abline(lm_female_2, col = "magenta")

# Adding a legend to differentiate between the points for male and female residuals
legend("topright", legend=c("Male", "Female", "Male Regr. Line", "Female Regr. Line"), col=c("gray", "red", "blue", "magenta"), pch=1, lty=1)

# (d) Calculating the mean and standard deviation of your statistic (sample variance) for each sample size
mean_var_20 <- mean(sample_variances)
sd_var_20 <- sd(sample_variances)
mean_var_100 <- mean(sample_variances_100)
sd_var_100 <- sd(sample_variances_100)
mean_var_10000 <- mean(sample_variances_10000)
sd_var_10000 <- sd(sample_variances_10000)

# Output the mean and standard deviation
cat("Mean Variance for 20 dice rolls: ", mean_var_20, "\n")
cat("Standard Deviation for 20 dice rolls: ", sd_var_20, "\n")
cat("Mean Variance for 100 dice rolls: ", mean_var_100, "\n")
cat("Standard Deviation for 100 dice rolls: ", sd_var_100, "\n")
cat("Mean Variance for 10000 dice rolls: ", mean_var_10000, "\n")
cat("Standard Deviation for 10000 dice rolls: ", sd_var_10000, "\n")

# (e) Discuss how the mean and the standard deviation change as the sample size increases
# In pdf
