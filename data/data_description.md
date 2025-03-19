1. **Fatalities (AER package)**
   - **Coverage**: 48 US states (1982–1988)  
   - **Variables**:  
     - `fatal`: number of traffic deaths  
     - `pop`: state population  
     - `drinkage`: minimum legal drinking age  
     - `beertax`: tax on a case of beer  
     - `miles`: vehicle miles traveled  
     - `unemp`: unemployment rate  
     - `income`: per capita personal income  
     - `jail/service`: DUI punishment indicators  
   - **Purpose**: Analyze how policy variables (e.g., beer taxes, legal drinking age) and economic factors influence traffic fatality rates.

2. **Grunfeld (plm package)**
   - **Coverage**: 11 large US manufacturing firms (1935–1954)  
   - **Variables**:  
     - `invest`: gross investment in plant & equipment (deflated)  
     - `value`: market value of the firm (equity plus debt)  
     - `capital`: deflated stock of plant & equipment  
   - **Purpose**: Standard panel data example for studying firm investment behavior over time; widely used in econometrics textbooks.

3. **WageData (panelr package)**
   - **Coverage**: 595 individuals, each observed in 7 time periods (balanced panel)  
   - **Variables**:  
     - `exp`: work experience  
     - `wks`: weeks worked  
     - `occ`: occupation code  
     - `ind`: industry code  
     - `south`, `smsa`, `ms`, `fem`, `union`, `ed`, `blk`  
     - `lwage`: log of wages  
     - `t`: time wave  
     - `id`: individual identifier  
   - **Purpose**: Illustrates panel data approaches in labor economics (e.g., wage equations with fixed effects).

4. **Gasoline (Ecdat package)**
   - **Coverage**: 18 OECD countries (1960–1978), 342 observations  
   - **Variables**:  
     - `country`, `year`  
     - `lgaspcar`: log of motor gasoline consumption per car  
     - `lincomep`: log of real per-capita income  
     - `lrpmg`: log of real motor gasoline price  
     - `lcarpcap`: log of cars per capita  
   - **Purpose**: Investigates gasoline demand and price/income elasticities across countries; showcases cross-country heterogeneity.

5. **Crime (Ecdat package)**
   - **Coverage**: 90 North Carolina counties (1981–1987), 630 observations  
   - **Variables**:  
     - `crmrte`: crime rate  
     - `prbarr`: probability of arrest  
     - `prbconv`: probability of conviction  
     - `prbpris`: probability of prison sentence  
     - `avgsen`: average sentence length (days)  
     - `polpc`: police per capita  
     - `density`, `taxpc`, `region`, `smsa`, `pctmin`  
   - **Purpose**: Evaluate how law enforcement and socioeconomic factors deter (or correlate with) crime rates over time.

6. **CigarettesSW (AER package)**
   - **Coverage**: 48 US states (1985 & 1995)  
   - **Variables**:  
     - `state`, `year`  
     - `cpi`: consumer price index  
     - `population`  
     - `packs`: packs of cigarettes per capita  
     - `income`: total state income (nominal)  
     - `tax`: combined excise taxes (state, local, federal)  
     - `price`: average cigarette price (incl. sales tax)  
   - **Purpose**: Examine effects of taxes/prices on cigarette consumption; frequently used in elasticity estimations.

7. **hprice2 (wooldridge package)**
   - **Coverage**: 506 housing observations (Boston area)  
   - **Variables**:  
     - `price`: median housing price  
     - `crime`: crimes per capita  
     - `nox`: nitrogen oxide concentration  
     - `rooms`: average number of rooms  
     - `dist`: weighted distance to employment centers  
     - `radial`: index of radial highways  
     - `proptax`: property tax per $1,000  
     - `stratio`: student-teacher ratio  
     - `lowstat`: percentage of lower-status population  
     - `lprice`, `lnox`, `lproptax`: log transformations  
   - **Purpose**: Used in hedonic price modeling and environmental economics to link housing prices with local amenities/disamenities.

8. **RDPerfComp (pder package)**
   - **Coverage**: 181 UK firms (1982–1989), ~1,378 observations  
   - **Variables**:  
     - `emp`: employment  
     - `rd`: R&D expenditures  
     - `sales`: firm sales  
     - `eps`: earnings per share  
     - `dpc`: dividends per share  
     - `cashflow`: operating cash flow  
     - `sector`: industrial classification  
   - **Purpose**: Demonstrates dynamic panel data methods for analyzing how R&D intensity affects firm performance over time.

9. **PSID (psidR package)**
   - **Coverage**: Panel Study of Income Dynamics (1968–present), ongoing US household survey  
   - **Variables**:  
     - Income measures (family, individual), demographic info, family structure  
   - **Purpose**: Longitudinal analysis of income, employment, and family dynamics; key for intergenerational mobility research.

10. **mpdta (did package)**
   - **Coverage**: 500 counties (2003–2007)  
   - **Variables**:  
     - `year`, `countyreal`  
     - `lpop`: log of population  
     - `lemp`: log of teen employment rate  
     - `first.treat`: first treatment year (0 if never treated)  
     - `treat`: treatment indicator  
   - **Purpose**: Illustrates staggered Difference-in-Differences analysis of minimum wage policies on teen employment.

11. **Scurvy (medicaldata package)**
   - **Coverage**: Historic clinical trial from 1747 (6 treatments × 2 doses, 12 observations)  
   - **Variables**:  
     - `treatment`: type of citrus or control  
     - `dose`: daily dose (oz)  
     - `gum_rot_d6`, `skin_sores_d6`, `strength_d6`, `fit_for_duty_d6`: scurvy-related outcomes  
   - **Purpose**: Famous small-sample data; now often used to demonstrate modern DiD approaches in a concise historical example.

12. **Medicaid Expansion Data**
   - **Coverage**: US states (1999–2014), differing Medicaid expansion adoption years  
   - **Variables**:  
     - `state`, `year`, `dins`: insurance rate  
     - `adopt_year`: year Medicaid expansion was adopted  
     - `treated`: expansion indicator  
     - `floor`: eligibility threshold  
     - `perc_elig`: percent of population eligible  
   - **Purpose**: Examine staggered policy adoption effects on insurance coverage and health outcomes; prominent in health economics DiD research.
