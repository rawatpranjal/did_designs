###############################################################################
# data.R : Single Script to Download & Save Various Panel/DiD Datasets
###############################################################################

# 0) Setup: Create "data" folder and install/load required packages
DATA_DIR <- "data"
if (!dir.exists(DATA_DIR)) {
  dir.create(DATA_DIR, showWarnings = FALSE, recursive = TRUE)
}

needed_pkgs <- c(
  "AER", "plm", "panelr", "Ecdat", "tidysynth",
  "devtools", "readr", "haven", 
  "dplyr", "stringr"
)

# Install missing packages quietly
to_install <- needed_pkgs[!(needed_pkgs %in% installed.packages()[, "Package"])]
if (length(to_install) > 0) {
  install.packages(to_install, quiet = TRUE)
}

# Install 'did' from GitHub if not present
if (!requireNamespace("did", quietly = TRUE)) {
  suppressMessages(devtools::install_github("bcallaway11/did", quiet = TRUE))
}

# Suppress library startup messages in one block
suppressPackageStartupMessages({
  invisible(lapply(c(needed_pkgs, "did"), function(pkg) {
    library(pkg, character.only = TRUE)
  }))
})

# Helper function to standardize CSV export
save_as_csv <- function(data, filename) {
  write.csv(data, file = file.path(DATA_DIR, filename), row.names = FALSE)
}

###############################################################################
# Unified List of Panel/DiD Datasets
###############################################################################

# 1) Fatalities (AER)
###############################################################################
# **Fatalities (AER package)**
# **Coverage**: 48 US states (1982–1988)  
# **Variables**:  
#  - `fatal`: number of traffic deaths  
#  - `pop`: state population  
#  - `drinkage`: minimum legal drinking age  
#  - `beertax`: tax on a case of beer  
#  - `miles`: vehicle miles traveled  
#  - `unemp`: unemployment rate  
#  - `income`: per capita personal income  
#  - `jail/service`: DUI punishment indicators  
# **Purpose**: Analyze how policy variables (e.g., beer taxes, legal drinking age) 
# and economic factors influence traffic fatality rates.
###############################################################################
data("Fatalities")
Fatalities_panel <- pdata.frame(Fatalities, index = c("state", "year"))
save_as_csv(Fatalities_panel, "fatalities.csv")

# 2) Grunfeld (plm)
###############################################################################
# **Grunfeld (plm package)**
# **Coverage**: 11 large US manufacturing firms (1935–1954)  
# **Variables**:  
#  - `invest`: gross investment in plant & equipment (deflated)  
#  - `value`: market value of the firm (equity plus debt)  
#  - `capital`: deflated stock of plant & equipment  
# **Purpose**: Standard panel data example for studying firm investment behavior 
# over time; widely used in econometrics textbooks.
###############################################################################
data("Grunfeld")
Grunfeld_panel <- pdata.frame(Grunfeld, index = c("firm", "year"))
save_as_csv(Grunfeld_panel, "grunfeld.csv")

# 3) WageData (panelr)
###############################################################################
# **WageData (panelr package)**
# **Coverage**: 595 individuals, each observed in 7 time periods (balanced panel)  
# **Variables**:  
#  - `exp`: work experience  
#  - `wks`: weeks worked  
#  - `occ`: occupation code  
#  - `ind`: industry code  
#  - `south`, `smsa`, `ms`, `fem`, `union`, `ed`, `blk`  
#  - `lwage`: log of wages  
#  - `t`: time wave  
#  - `id`: individual identifier  
# **Purpose**: Illustrates panel data approaches in labor economics 
# (e.g., wage equations with fixed effects).
###############################################################################
data("WageData")
wages_panel <- panel_data(WageData, id = id, wave = t)
save_as_csv(as.data.frame(wages_panel), "wagedata.csv")

# 4) Gasoline (Ecdat)
###############################################################################
# **Gasoline (Ecdat package)**
# **Coverage**: 18 OECD countries (1960–1978), 342 observations  
# **Variables**:  
#  - `country`, `year`  
#  - `lgaspcar`: log of motor gasoline consumption per car  
#  - `lincomep`: log of real per-capita income  
#  - `lrpmg`: log of real motor gasoline price  
#  - `lcarpcap`: log of cars per capita  
# **Purpose**: Investigates gasoline demand and price/income elasticities 
# across countries; showcases cross-country heterogeneity.
###############################################################################
data("Gasoline")
gasoline_panel <- pdata.frame(Gasoline, index = c("country", "year"))
save_as_csv(gasoline_panel, "gasoline.csv")

# 5) Crime (Ecdat)
###############################################################################
# **Crime (Ecdat package)**
# **Coverage**: 90 North Carolina counties (1981–1987), 630 observations  
# **Variables**:  
#  - `crmrte`: crime rate  
#  - `prbarr`: probability of arrest  
#  - `prbconv`: probability of conviction  
#  - `prbpris`: probability of prison sentence  
#  - `avgsen`: average sentence length (days)  
#  - `polpc`: police per capita  
#  - `density`, `taxpc`, `region`, `smsa`, `pctmin`  
# **Purpose**: Evaluate how law enforcement and socioeconomic factors deter 
# (or correlate with) crime rates over time.
###############################################################################
data("Crime")
crime_panel <- pdata.frame(Crime, index = c("county", "year"))
save_as_csv(crime_panel, "crime.csv")

# 6) CigarettesSW (AER)
###############################################################################
# **CigarettesSW (AER package)**
# **Coverage**: 48 US states (1985 & 1995)  
# **Variables**:  
#  - `state`, `year`  
#  - `cpi`: consumer price index  
#  - `population`  
#  - `packs`: packs of cigarettes per capita  
#  - `income`: total state income (nominal)  
#  - `tax`: combined excise taxes (state, local, federal)  
#  - `price`: average cigarette price (incl. sales tax)  
# **Purpose**: Examine effects of taxes/prices on cigarette consumption; 
# frequently used in elasticity estimations.
###############################################################################
data("CigarettesSW")
cigarettes_panel <- pdata.frame(CigarettesSW, index = c("state", "year"))
save_as_csv(cigarettes_panel, "cigarettes.csv")

# 7) RDPerfComp (pder)
###############################################################################
# **RDPerfComp (pder package)**
# **Coverage**: 181 UK firms (1982–1989), ~1,378 observations  
# **Variables**:  
#  - `emp`: employment  
#  - `rd`: R&D expenditures  
#  - `sales`: firm sales  
#  - `eps`: earnings per share  
#  - `dpc`: dividends per share  
#  - `cashflow`: operating cash flow  
#  - `sector`: industrial classification  
# **Purpose**: Demonstrates dynamic panel data methods for analyzing 
# how R&D intensity affects firm performance over time.
###############################################################################
if (require("pder", quietly = TRUE)) {
  data("RDPerfComp")
  rdperf_panel <- pdata.frame(RDPerfComp, index = c("id", "year"))
  save_as_csv(rdperf_panel, "rdperfcomp.csv")
}

# 8) Medicaid Expansion (Staggered Adoption)
###############################################################################
# **Medicaid Expansion Data**
# **Coverage**: US states (1999–2014), differing Medicaid expansion adoption years  
# **Variables**:  
#  - `state`, `year`, `dins`: insurance rate  
#  - `adopt_year`: year Medicaid expansion was adopted  
#  - `treated`: expansion indicator  
#  - `floor`: eligibility threshold  
#  - `perc_elig`: percent of population eligible  
# **Purpose**: Examine staggered policy adoption effects on insurance coverage 
# and health outcomes; prominent in health economics DiD research.
###############################################################################
df_medicaid <- read_dta(
  "https://raw.githubusercontent.com/Mixtape-Sessions/Advanced-DID/main/Exercises/Data/ehec_data.dta"
)
save_as_csv(df_medicaid, "medicaid.csv")

# 9) California Smoking (tidysynth)
###############################################################################
# **California Smoking (tidysynth package)**  
# **Coverage**: 39 US states (1970–2000)  
# **Variables**:  
#  - `cigsale`: Cigarette sales per 100,000 people  
#  - `retprice`: Adjusted retail cigarette price  
#  - `lnincome`: Log mean income  
#  - `age15to24`: % population aged 15–24  
#  - `beer`: Beer sales per capita  
# **Purpose**: Benchmark for synthetic control methods to evaluate 
# California's 1988 tobacco tax (Proposition 99) on cigarette consumption.
###############################################################################
data("smoking")
smoking_panel <- pdata.frame(smoking, index = c("state", "year"))
save_as_csv(smoking_panel, "smoking.csv")

# 10) Card & Krueger Fast-Food Data
###############################################################################
# **Card & Krueger Fast-Food Data**
# **Coverage**: Fast food restaurants in NJ and PA (1992-1993)
# **Variables**:
#  - `sheet`: Identifier
#  - `chain`: Restaurant chain 
#  - `state`: State indicator (NJ/PA)
#  - `southj`: South Jersey indicator
#  - `centralj`: Central Jersey indicator
#  - `northj`: North Jersey indicator
#  - `pa`: Pennsylvania indicator
#  - `shore`: Shore area indicator
#  - `price variables`: Various price measures
#  - `emptot`: Total full-time equivalent employees
#  - `wage variables`: Starting wages and other labor cost measures
# **Purpose**: Famous dataset for DiD analysis of minimum wage effects on
# employment, from Card & Krueger's landmark 1994 study.
###############################################################################
tempfile_path <- tempfile()
download.file("http://davidcard.berkeley.edu/data_sets/njmin.zip", destfile = tempfile_path, quiet = TRUE)
tempdir_path <- tempdir()
unzip(tempfile_path, exdir = tempdir_path)

codebook_lines <- readr::read_lines(file = paste0(tempdir_path, "/codebook"))

variable_names <- codebook_lines %>%
  .[8:59] %>%
  .[-c(5, 6, 13, 14, 32, 33)] %>%
  str_sub(1, 13) %>%
  str_squish() %>%
  str_to_lower()

fastfood_raw <- read_table(  # replaced read_table2() with read_table()
  file = paste0(tempdir_path, "/public.dat"),
  col_names = FALSE
)

fastfood_df <- fastfood_raw %>%
  select(-X47) %>%
  `colnames<-`(variable_names) %>%
  mutate_all(as.numeric) %>%
  mutate(sheet = as.character(sheet))

save_as_csv(fastfood_df, "fast_food.csv")

# 11) mpdta (did)
###############################################################################
# **mpdta (Minimum Wage Dataset)**
# **Coverage**: 500 US counties (2003–2007)
# **Key Variables**:
#  - `lpop`: Log of county population
#  - `lemp`: Log of teen employment rate
#  - `first.treat`: First treatment year (0 = never treated)
#  - `treat`: Treatment indicator
#  - `countyreal`: County identifier
#  - `year`: Observation year
# **Purpose**: Standard dataset for staggered DiD applications, studying 
# minimum wage policy impacts on teen employment. Featured in 
# Callaway & Sant'Anna (2020).
###############################################################################
data("mpdta", package = "did")
save_as_csv(mpdta, "mpdta.csv")

# Final message (minimal console prints)
message("All panel/DiD datasets have been saved to the '", DATA_DIR, "' folder.")