# 1️⃣ Load necessary packages
library(dplyr)      # data wrangling
library(tidyr)      # data cleaning
library(readr)      # read CSV/Excel
library(ggplot2)    # visualization
library(car)        # VIF, diagnostics
library(lmtest)     # heteroskedasticity tests
library(sandwich)   # robust SE
library(broom)  
library(stargazer)
# Optional for panel data / fixed effects
library(plm)        # panel data models
library(fixest)     # fast FE model

house_prob_moderate_rooms <- read.csv('C:/Users/nph1u24/OneDrive - University of Southampton/MSc Bussiness Analytics and Finance/Project/Data/UK House price/houses_final.csv')
View(house_prob_moderate_rooms)
df <- house_prob_moderate_rooms %>%select('Transaction_ID',
                                          'House_Price_w', 'Property_type',
                                          'Old_New','year_trans',
                                          'Region', 'duration',
                                          'tfarea_w',
                                          'numberrooms_w', 'nearest_station',
                                          'nearest_university_dist',
                                          'nearest_green',
                                          'nearest_water_dist',
                                          'rf_flood',
                                          'log_flood') 

# Suppose your data frame is called df
cols_to_factor <- c("Property_type",
                    "Old_New",
                    "year_trans",
                    "Region",
                    "duration",
                    'rf_flood',
                    'log_flood')

# Convert selected columns to factors
df[cols_to_factor] <- lapply(df[cols_to_factor], as.factor)



#summary
stargazer(df)




# --- Formula (same predictors) ---
form1 <- as.formula(
  log(House_Price_w) ~ Property_type + Old_New + duration + Region + year_trans +
    tfarea_w + numberrooms_w +
    nearest_station + nearest_university_dist +
    nearest_green + nearest_water_dist + rf_flood
)

form2 <- as.formula(
  log(House_Price_w) ~ Property_type + Old_New + duration + Region + year_trans +
    tfarea_w + numberrooms_w +
    nearest_station + nearest_university_dist +
    nearest_green + nearest_water_dist + log_flood
)



# Run hedonic model (log-price as dependent variable)
model <- lm(form1, data = df, na.action = na.omit)

# Summary of results
summary(model)

#crPlots(model)

vif(model)   # Variance Inflation Factor

bptest(model) #heteroskedasticity
# Robust standard errors (HC1 is a common choice)
coeftest(model, vcov = vcovHC(model, type = "HC3"))
# HC3 covariance
rob_vcov <- vcovHC(model, type = "HC3")
# Robust Type-III Wald tests (joint tests per term, incl. factor blocks)
Anova(model, type = "III", vcov. = rob_vcov) 

# Calculate standard errors clustered by the 'Region' variable
coeftest(model, vcov = vcovCL(model, cluster = ~ Region))

#Linearity
#plot(model, which = 1)
#functional form misspecification
resettest(model)

#shapiro.test(residuals(model))   # Shapiro-Wilk test residual normality
#qqnorm(residuals(model)); qqline(residuals(model))

dwtest(model)   # Durbin-Watson test autocorellation
#plot fitted vs. residuals directly to check functional form
plot(fitted(model), resid(model))
abline(h = 0, col = "red")

#Latex table
stargazer(model, title="Results", align=TRUE)
