# Load necessary libraries
library(dplyr)
library(tidyr)
library(lmtest)
library(sandwich)

# Read the medicaid.csv file
medicaid_data <- read.csv("data/medicaid.csv")

# Explore the data structure
str(medicaid_data)
head(medicaid_data)

# Prepare the dataset for 2×2 DiD analysis
did_data <- medicaid_data %>%
  # Create treatment group indicator (states that expanded in 2014)
  mutate(
    expansion = ifelse(!is.na(yexp2) & yexp2 == 2014, 1, 0),
    # Only keep non-expansion states (NA or year > 2019) and 2014 expansion states
    keep = ifelse(is.na(yexp2) | yexp2 > 2019 | yexp2 == 2014, 1, 0)
  ) %>%
  # Filter to keep only control group and 2014 expansion group
  filter(keep == 1) %>%
  # Keep only years 2013 and 2014 for the 2×2 analysis
  filter(year %in% c(2013, 2014)) %>%
  # Create post indicator
  mutate(post = ifelse(year == 2014, 1, 0))

# Check the distribution of our treatment and control groups
table(did_data$expansion)

# Verify we have the right years
table(did_data$year)

# Assuming 'dins' is the mortality rate variable
# Rename for clarity
did_data <- did_data %>%
  rename(mortality = dins)

# -----------------------------------------
# Replicate Table 2: Simple 2×2 DiD
# -----------------------------------------

# Function to compute the means and DiD
compute_did_table <- function(data, weighted = FALSE) {
  if (weighted) {
    # Weighted means
    means <- data %>%
      group_by(expansion, year) %>%
      summarize(
        mortality = weighted.mean(mortality, w = W),
        .groups = 'drop'
      ) %>%
      pivot_wider(names_from = year, values_from = mortality, names_prefix = "y") %>%
      mutate(trend = y2014 - y2013)
  } else {
    # Unweighted means
    means <- data %>%
      group_by(expansion, year) %>%
      summarize(
        mortality = mean(mortality),
        .groups = 'drop'
      ) %>%
      pivot_wider(names_from = year, values_from = mortality, names_prefix = "y") %>%
      mutate(trend = y2014 - y2013)
  }
  
  # Calculate the gap between groups
  gap_2013 <- means$y2013[means$expansion == 1] - means$y2013[means$expansion == 0]
  gap_2014 <- means$y2014[means$expansion == 1] - means$y2014[means$expansion == 0]
  
  # Calculate DiD
  did <- gap_2014 - gap_2013
  # or equivalently:
  # did <- means$trend[means$expansion == 1] - means$trend[means$expansion == 0]
  
  # Format results for the table
  result <- data.frame(
    Group = c("Expansion", "No Expansion", "Gap"),
    Y2013 = c(means$y2013[means$expansion == 1], means$y2013[means$expansion == 0], gap_2013),
    Y2014 = c(means$y2014[means$expansion == 1], means$y2014[means$expansion == 0], gap_2014),
    Trend = c(means$trend[means$expansion == 1], means$trend[means$expansion == 0], did)
  )
  
  return(result)
}

# Compute the unweighted and weighted tables
table2_unweighted <- compute_did_table(did_data, weighted = FALSE)
table2_weighted <- compute_did_table(did_data, weighted = TRUE)

# Print Table 2
print("Table 2: Simple 2×2 DiD")
print("Unweighted Averages:")
print(round(table2_unweighted, 3))
print("Weighted Averages:")
print(round(table2_weighted, 3))

# -----------------------------------------
# Replicate Table 3: Regression 2×2 DiD
# -----------------------------------------

# Function to run DiD regressions and extract coefficients
run_did_regressions <- function(data, weighted = FALSE) {
  # Prepare data for regressions
  data$interaction <- data$expansion * data$post
  
  # For the differenced regression
  data_diff <- data %>%
    pivot_wider(
      id_cols = c(stfips, expansion, W),
      names_from = year,
      values_from = mortality
    ) %>%
    mutate(mortality_diff = `2014` - `2013`)
  
  # Weights for weighted regressions (or NULL for unweighted)
  weights_var <- if(weighted) data$W else NULL
  weights_var_diff <- if(weighted) data_diff$W else NULL
  
  # Model 1: Basic DiD regression
  model1 <- lm(mortality ~ expansion + post + interaction, 
               data = data, 
               weights = weights_var)
  
  # Model 2: DiD with fixed effects
  model2 <- lm(mortality ~ interaction + factor(stfips) + factor(year),
               data = data,
               weights = weights_var)
  
  # Model 3: Differenced regression (first differences)
  model3 <- lm(mortality_diff ~ expansion,
               data = data_diff,
               weights = weights_var_diff)
  
  # Get clustered SEs for all models
  # Cluster at the state level
  se1 <- sqrt(diag(vcovCL(model1, cluster = data$stfips)))
  se2 <- sqrt(diag(vcovCL(model2, cluster = data$stfips)))
  se3 <- sqrt(diag(vcovCL(model3, cluster = data_diff$stfips)))
  
  # Extract coefficients and SEs
  results <- data.frame(
    Model = c("Model 1", "Model 1", "Model 1", "Model 1", 
              "Model 2", "Model 3", "Model 3"),
    Variable = c("Constant", "Medicaid Expansion", "Post", "Medicaid Expansion × Post", 
                 "Medicaid Expansion × Post", "Constant", "Medicaid Expansion"),
    Coefficient = c(coef(model1)["(Intercept)"], 
                    coef(model1)["expansion"], 
                    coef(model1)["post"], 
                    coef(model1)["interaction"],
                    coef(model2)["interaction"],
                    coef(model3)["(Intercept)"],
                    coef(model3)["expansion"]),
    SE = c(se1["(Intercept)"], 
           se1["expansion"], 
           se1["post"], 
           se1["interaction"],
           se2["interaction"],
           se3["(Intercept)"],
           se3["expansion"])
  )
  
  return(results)
}

# Run regressions with and without weights
table3_unweighted <- run_did_regressions(did_data, weighted = FALSE)
table3_weighted <- run_did_regressions(did_data, weighted = TRUE)

# Print Table 3
print("Table 3: Regression 2×2 DiD")
print("Unweighted Regressions:")
print(table3_unweighted)
print("Weighted Regressions:")
print(table3_weighted)

# Create a nicely formatted version of Table 3
format_table3 <- function(unweighted, weighted) {
  # Prepare formatted strings with asterisks for significance
  format_coef <- function(coef, se) {
    significance <- ""
    if (abs(coef/se) > 2.58) significance <- "***"
    else if (abs(coef/se) > 1.96) significance <- "**"
    else if (abs(coef/se) > 1.65) significance <- "*"
    
    return(paste0(round(coef, 3), significance, " (", round(se, 3), ")"))
  }
  
  # Format all coefficients
  unweighted$formatted <- mapply(format_coef, unweighted$Coefficient, unweighted$SE)
  weighted$formatted <- mapply(format_coef, weighted$Coefficient, weighted$SE)
  
  # Construct the table
  table3 <- data.frame(
    Variable = c("Constant", "Medicaid Expansion", "Post", "Medicaid Expansion × Post", 
                 "State fixed effects", "Year fixed effects"),
    Model1_unw = c(unweighted$formatted[unweighted$Variable == "Constant"],
                   unweighted$formatted[unweighted$Variable == "Medicaid Expansion"],
                   unweighted$formatted[unweighted$Variable == "Post"],
                   unweighted$formatted[unweighted$Variable == "Medicaid Expansion × Post"],
                   "No", "No"),
    Model2_unw = c("", "", "", 
                   unweighted$formatted[unweighted$Variable == "Medicaid Expansion × Post" & 
                                        unweighted$Model == "Model 2"],
                   "Yes", "Yes"),
    Model3_unw = c(unweighted$formatted[unweighted$Variable == "Constant" & 
                                       unweighted$Model == "Model 3"],
                   unweighted$formatted[unweighted$Variable == "Medicaid Expansion" & 
                                       unweighted$Model == "Model 3"],
                   "", "", "No", "No"),
    Model1_w = c(weighted$formatted[weighted$Variable == "Constant"],
                 weighted$formatted[weighted$Variable == "Medicaid Expansion"],
                 weighted$formatted[weighted$Variable == "Post"],
                 weighted$formatted[weighted$Variable == "Medicaid Expansion × Post"],
                 "No", "No"),
    Model2_w = c("", "", "",
                weighted$formatted[weighted$Variable == "Medicaid Expansion × Post" & 
                                   weighted$Model == "Model 2"],
                "Yes", "Yes"),
    Model3_w = c(weighted$formatted[weighted$Variable == "Constant" & 
                                  weighted$Model == "Model 3"],
                weighted$formatted[weighted$Variable == "Medicaid Expansion" & 
                                  weighted$Model == "Model 3"],
                "", "", "No", "No")
  )
  
  colnames(table3) <- c("Variable", 
                        "Unweighted (1)", "Unweighted (2)", "Unweighted (3)",
                        "Weighted (4)", "Weighted (5)", "Weighted (6)")
  
  return(table3)
}

# Create the formatted Table 3
table3_formatted <- format_table3(table3_unweighted, table3_weighted)
print(table3_formatted)

# If the results don't match the paper exactly, print some summary statistics to understand why
print("Summary of mortality variable:")
summary(did_data$mortality)

print("Sample sizes by group:")
table(did_data$expansion, did_data$year)

# Compare our estimates to the paper's values
paper_table2_unweighted <- data.frame(
  Group = c("Expansion", "No Expansion", "Gap/DiD"),
  Y2013 = c(419.2, 474.0, -54.8),
  Y2014 = c(428.5, 483.1, -54.7),
  Trend = c(9.3, 9.1, 0.1)
)

paper_table2_weighted <- data.frame(
  Group = c("Expansion", "No Expansion", "Gap/DiD"),
  Y2013 = c(322.7, 376.4, -53.7),
  Y2014 = c(326.5, 382.7, -56.2),
  Trend = c(3.7, 6.3, -2.6)
)

print("Comparison with paper values (Table 2 unweighted):")
print(paper_table2_unweighted)
print("Our estimates (Table 2 unweighted):")
print(table2_unweighted)

print("Comparison with paper values (Table 2 weighted):")
print(paper_table2_weighted)
print("Our estimates (Table 2 weighted):")
print(table2_weighted)