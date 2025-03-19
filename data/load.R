# Set output directory to data subfolder
output_dir <- "/Users/pranjal/Code/did_designs/data"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Install required packages
install.packages(c("AER", "plm", "panelr", "psidR", "Ecdat", "wooldridge", "pder", "TeachingDemos","wooldridge"))
devtools::install_github("bcallaway11/did")

library(AER)
library(plm)
library(panelr)

# 1. Traffic Fatalities (AER package) - 1982-1988
data("Fatalities")
Fatalities_panel <- pdata.frame(Fatalities, index = c("state", "year"))
write.csv(Fatalities_panel, file.path(output_dir, "fatalities_panel.csv"), row.names = FALSE)

# 2. Investment Data (plm package) - 1935-1954
data("Grunfeld")
Grunfeld_panel <- pdata.frame(Grunfeld, index = c("firm", "year")) 
write.csv(Grunfeld_panel, file.path(output_dir, "grunfeld_panel.csv"), row.names = FALSE)

# 3. Wage Data (panelr package) - 1976-1982
data("WageData")
wages_panel <- panel_data(WageData, id = id, wave = t)

# Convert panel_data object to regular data frame first
wages_df <- as.data.frame(wages_panel)

# Then write to CSV
write.csv(wages_df, file.path(output_dir, "wagedata_panel.csv"), row.names = FALSE)

# 4. Gasoline Demand Data (plm package) - 1960-1978
library(Ecdat)
data("Gasoline")
gasoline_panel <- pdata.frame(Gasoline, index = c("country", "year"))
write.csv(gasoline_panel, file.path(output_dir, "gasoline_panel.csv"), row.names = FALSE)

# 5. Crime Data (Ecdat package) - 1981-1987
data("Crime")
crime_panel <- pdata.frame(Crime, index = c("county", "year"))
write.csv(crime_panel, file.path(output_dir, "crime_panel.csv"), row.names = FALSE)

# 6. Cigarette Consumption Data (AER package) - 1985-1995
data("CigarettesSW")
cigarettes_panel <- pdata.frame(CigarettesSW, index = c("state", "year"))
write.csv(cigarettes_panel, file.path(output_dir, "cigarettes_panel.csv"), row.names = FALSE)

# 7. Hedonic Housing Prices (wooldridge package) - 1978 study
if(require(wooldridge)) {
  data("hprice2")
  hedonic_panel <- pdata.frame(hprice2, index = c("radial", "dist"))
  write.csv(hedonic_panel, file.path(output_dir, "hedonic_panel.csv"), row.names = FALSE)
}

# 8. R&D Performance Dataset (pder package) - 1982-1989
if(require(pder)) {
  data("RDPerfComp")
  rdperf_panel <- pdata.frame(RDPerfComp, index = c("id", "year"))
  write.csv(rdperf_panel, file.path(output_dir, "rdperf_panel.csv"), row.names = FALSE)
}


# PSID Data Download (requires credentials)
if(FALSE) {  # Change to TRUE to run
  library(psidR)
  # Early PSID waves - 1968-1972
  early_fam_vars <- data.frame(
    year = c(1968, 1969, 1970, 1971, 1972),
    income = c("V118", "V1514", "V2226", "V2852", "V3256")
  )
  early_psid_data <- build.psid(
    datadir = file.path(output_dir, "psid_early_raw"),
    fam.vars = early_fam_vars
  )
  write.csv(early_psid_data, file.path(output_dir, "psid_early_panel.csv"), row.names = FALSE)
  
  # Recent PSID waves
  recent_fam_vars <- data.frame(
    year = c(2013, 2015, 2017),
    income = c("ER58223", "ER60463", "ER66031")
  )
  recent_psid_data <- build.psid(
    datadir = file.path(output_dir, "psid_recent_raw"),
    fam.vars = recent_fam_vars
  )
  write.csv(recent_psid_data, file.path(output_dir, "psid_recent_panel.csv"), row.names = FALSE)
}

# Load the package
library(did)
data(mpdta)
write.csv(mpdta, "/Users/pranjal/Code/did_designs/data/mpdta.csv", row.names = FALSE)

