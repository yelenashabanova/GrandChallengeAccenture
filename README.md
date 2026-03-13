# GrandChallengeAccenture

# 1. Introduction


Air pollution is one of the most significant environmental and public health challenges in Europe. Among the different pollutants monitored by environmental agencies, fine particulate matter (PM2.5) is particularly harmful because it can penetrate deep into the lungs and bloodstream.

The main goal of the full project is to predict future Air Quality Index (AQI) values or flag regulatory limit exceedances using machine learning across European countries, with a particular focus on Italy, in order to understand the patterns and potential drivers of PM2.5 pollution.

Analyse pollutant levels in European cities using 10 years of annual air quality statistics (2015–2024) published by the European Environment Agency (EEA).

The insights we got by this analysis are particularly relevant for different audiences:

- Public health authorities
- Environmental protection agencies
- Urban policy makers

These stakeholders rely on air quality monitoring data to identify pollution hotspots, evaluate environmental policies, and design strategies to reduce population exposure to harmful pollutants.

Our goal is therefore not only to explore the dataset, but also to move toward identifying factors that could explain and eventually predict PM2.5 pollution levels. Here we are going to explain our decisions based on the analyses, which led to picking this specific pollutant.

---

# 2. Data Flaws and Data Story


Firstly, we had to analyze the data provided. The main datasets used in the project include:

## 2.1 Firstly, we analyzed the raw data provided

The core dataset comes from the EEA Annual Air Quality Statistics (e-Reporting) and covers the period 2015--2024. Instead of one single file, the data was provided as 10 separate CSV files, one for each year, which we then combined into one main DataFrame.

After concatenation, the dataset contains:

- $468{,}481$ rows
- $27$ columns

Each row represents:

$$
\text{one monitoring station} \times \text{one pollutant} \times \text{one annual statistical summary} \times \text{one year}
$$

The rows are not raw sensor-by-sensor hourly measurements. They are already aggregated annual air quality statistics, such as annual means, percentiles, maxima, or daily-based summaries.

### Supporting outputs

[images/raw_data_head.png](images/raw_data_head.png)

[images/data_dimensions.png](images/data_dimensions.png)

## 2.2 What information was available in the raw dataset

From the raw data preview and column inspection, we identified several groups of variables.

### 2.2.1 Identification and network information

These columns describe the monitoring network and station identity:

- Country
- Air Quality Network
- Air Quality Network Name
- Air Quality Station EoI Code
- Air Quality Station Name
- Sampling Point Id

### 2.2.2 Pollutant information

These columns tell us what pollutant is being measured:

- Air Pollutant
- Air Pollutant Description
- Unit Of Air Pollution Level

The raw dataset contains $38$ pollutant labels, although we later focused on the most relevant pollutants for analysis.

### 2.2.3 Aggregation / reporting information

These columns explain how the reported value was computed:

- Data Aggregation Process Id
- Data Aggregation Process

This is a key point of the dataset: each value corresponds to an annual aggregation process, not to a raw reading.

### 2.2.4 Measurement information

These are the main analytical variables:

- Year
- Air Pollution Level
- Data Coverage
- Verification

### 2.2.5 Spatial and contextual information

These variables describe the station location and setting:

- Air Quality Station Type
- Air Quality Station Area
- Longitude
- Latitude
- Altitude
- City

### Supporting outputs

[images/raw_data_columns.png](images/raw_data_columns.png)

[images/raw_data_describe.png](images/raw_data_describe.png)

## 2.3 Additional data used for comparison

Besides the main Italy-focused dataset, we also loaded a country comparison dataset for:

- France
- Switzerland
- Austria

This comparison subset was filtered to the focus pollutants and the selected aggregation process, and it was used to support cross-country comparisons later in the exploratory analysis.

This was useful because it allowed us to understand whether some observed pollution patterns were specific to Italy or also visible in nearby European countries.

### Supporting outputs

[images/country_comparison_counts.png](images/country_comparison_counts.png)

## 2.4 What we evaluated before the analysis

Before performing any analysis, we first evaluated the quality and structure of the dataset.

The first checks focused on:

- missing values
- data coverage
- verification flags
- duplicate rows
- the meaning of the aggregation process
- the availability of spatial fields such as city and coordinates

## 2.5 Main Data Issues Identified

After loading and inspecting the raw dataset, we did a series of data quality checks to evaluate whether it is relevant and reliable for analysis.

The evaluation focused on common issues in environmental monitoring data, such as:

- missing values
- duplicate observations
- invalid pollution measurements
- incomplete coverage of monitoring stations

### Missing Values

The analysis shows that while most fields are relatively complete, some important variables suffer from significant missingness.

The City variable is missing for a large share of observations ($62\%$), which limits our ability to perform precise geographic analysis.

Since understanding spatial pollution patterns is central to this project, recovering geographic information becomes a key step in the preprocessing pipeline.

### Supporting output

[images/missing_value_barchart.png](images/missing_value_barchart.png)

### Duplicate Observations

Next, we checked whether the dataset contained duplicate rows, which could bias statistical summaries or lead to double-counting of pollution measurements. The analysis confirms that no duplicate observations were detected, indicating that the dataset structure is internally consistent.

### Supporting output

[images/duplicate_check_result.png](images/duplicate_check_result.png)

### Invalid Pollution Values

Air Pollution Level is a key variable. Pollution concentrations should never take negative values. Therefore, we verified whether the dataset contained invalid measurements.

We recognized that a small number of negative values were present, which likely correspond to reporting errors. These observations were removed during preprocessing to ensure that all pollution measurements are physically meaningful.

### Supporting output

[images/negative_values_removed.png](images/negative_values_removed.png)

### Data Coverage Evaluation

Since environmental monitoring stations may not operate year-round, the dataset incorporates a Data Coverage variable. This variable indicates the percentage of time during which measurements were successfully recorded. We analyzed the distribution of this variable to determine whether the reported annual statistics are based on sufficiently complete data.

The results show that most observations have acceptable coverage, meaning that the annual pollution statistics can generally be considered reliable.

### Supporting output

[images/distribution_of_data_coverage.png](images/distribution_of_data_coverage.png)

### Aggregation Process

Finally, we analyzed the Data Aggregation Process variable: how the pollution value was computed, for example whether it represents:

- an annual mean
- a percentile value
- a maximum daily concentration

Because different aggregation methods describe different aspects of pollution exposure, it is important to understand which aggregation types dominate the dataset before comparing values across observations.

### Supporting output

[images/top_15_aggregation_processes.png](images/top_15_aggregation_processes.png)

## 2.6 What We Decided to Fix or Enrich

Based on the issues identified during the data quality evaluation, we implemented several adjustments to improve the dataset and ensure that the subsequent analysis would be meaningful.

### Recovering Missing Geographic Information (City)

One of the most significant data issues identified during the quality assessment was the large share of missing values in the City variable, which was missing for approximately $62\%$ of observations. Since geographic analysis is central to understanding pollution patterns, recovering city-level information became a critical preprocessing step.

To handle this issue, using spatial data from ISTAT municipalities, we were able to assign many stations to specific cities.

Using GeoPandas, we performed a spatial join between station coordinates and the ISTAT comuni shapefile, assigning each monitoring station to the municipality in which it is located. This allowed us to reconstruct missing city values for many stations and significantly improved the spatial interpretability of the dataset.

### Supporting outputs

[images/joined_comune.csv](images/joined_comune.csv)

[images/city_recovery_before_after.png](images/city_recovery_before_after.png)

### Cleaning Invalid Pollution Values

During the inspection of the Air Pollution Level variable, we found a small number of negative values, which are not physically meaningful for pollution concentrations. These values likely correspond to measurement or reporting errors. To ensure that all observations represent valid environmental measurements, negative pollution values were removed during preprocessing.

### Supporting outputs

[images/negative_values_removed.png](images/negative_values_removed.png)

### Evaluating Measurement Coverage

The Data Coverage variable represents the percentage of time during which measurements were recorded. We examined the distribution of this variable to determine whether annual pollution statistics are based on sufficiently complete measurements. The results show that most observations have acceptable coverage, meaning that the reported pollution statistics can generally be considered reliable.

### Supporting outputs

[images/data_coverage_distribution.png](images/data_coverage_distribution.png)

### Motivation for Additional Contextual Variables

Although the dataset provides detailed pollution measurements, it contains very limited contextual information about the environment surrounding monitoring stations.

In particular, important factors that may influence pollution levels --- geographic characteristics, meteorological conditions, and population distribution --- are either missing or static in the provided data.

For example, the population information included in the dataset does not vary across years, which limits its usefulness for analyzing demographic effects over time.

Because of these limitations, the dataset in its current form may not fully explain why pollution levels differ between locations. Therefore, the next step of the project is to investigate whether external contextual variables could help explain the observed spatial differences in pollution levels.

### Population Data Limitation

Another limitation identified during the data exploration concerns population information.

The population data provided in the dataset appears constant across all years, which prevents us from analyzing changes in population over time.

Because of this limitation, we plan to integrate an updated population dataset from ISTAT in the next stage of the project, which will allow us to build dynamic demographic information into future modeling. This additional dataset will enable us to better capture the relationship between population distribution and pollution exposure.

Therefore, the key question becomes whether contextual environmental variables can explain the spatial differences observed in pollution levels.

# 3. Initial Analytical Assumption

EDA Section 4 from notebook.

Pollutant distribution includes:

- pollutant frequency
- pollutant value distributions
- skewness
- outliers

The key question is: what is in the dataset, and which pollutant should we study?

After cleaning and preparing the dataset, the next step is to understand what insights can already be obtained from the available pollution measurements, before introducing additional contextual variables.

**Initial analytical assumption**

Pollution patterns observed in the dataset may potentially be explained using only the variables already present in the air quality monitoring data.

To evaluate this assumption, we first explore the structure of the pollution measurements themselves, focusing on three aspects:

- which pollutants dominate the dataset
- how pollution values are distributed
- whether meaningful patterns appear across pollutants, time, and geography

This initial exploration helps us determine whether the existing dataset is sufficient to explain pollution patterns or whether additional environmental variables may be necessary.

## 3.1 Distribution of Pollutants in the Dataset

Which pollutants are most frequently reported in the dataset? The results show that although the dataset contains many pollutant types, the majority of observations are concentrated in a smaller group of pollutants.

### Supporting output

[images/pollutant_counts.png](images/pollutant_counts.png)

This distribution indicates that a few pollutants dominate the monitoring data, particularly:

- PM2.5
- PM10
- NO$_2$
- O$_3$

These pollutants appear consistently across years and monitoring stations, making them the most suitable candidates for further analysis.

## 3.2 Distribution of Pollution Levels

Next, we analyzed the distribution of pollution concentrations for the main pollutants.

### Supporting outputs

[images/pollution_level_distributions.png](images/pollution_level_distributions.png)

[images/overall_pollution_level_distributions.png](images/overall_pollution_level_distributions.png)

The histograms show that pollution levels are strongly right-skewed, meaning that most monitoring stations report moderate pollution levels while a smaller number of locations experience very high concentrations.

The boxplots further confirm the presence of extreme pollution values, suggesting that pollution exposure may be concentrated in specific areas.

This observation provides an important initial insight:

Pollution levels are not evenly distributed across monitoring stations, which suggests that local environmental conditions may influence pollution concentrations.

## 3.3 Selecting the Focus Pollutant

Since the dataset contains several pollutants, it is necessary to understand which pollutant should become the main focus of the analysis.

Among the available pollutants, PM2.5 emerges as the most relevant candidate for several reasons:

First, PM2.5 is widely recognized as one of the most harmful air pollutants, due to its ability to penetrate deeply into the respiratory system and bloodstream.

Second, PM2.5 appears frequently in the monitoring dataset, ensuring sufficient observations for reliable analysis.

Third, PM2.5 is commonly used as a key indicator of air quality in environmental and public health research, making it particularly relevant for policy analysis.

For these reasons, PM2.5 was selected as the primary pollutant of interest for the subsequent exploratory analysis.

This choice is also supported by a chemical and health perspective on pollutant danger:

> From a chemical/health perspective, ``most dangerous'' is based on toxicity, exposure, and regulatory limits (WHO/EU). A common ranking is  
> $\mathbf{PM2.5} > \mathbf{PM10} > \mathbf{NO_2} > \mathbf{O_3} > \mathbf{SO_2}$.

Here:

- $\mathrm{PM2.5}$ = fine particles, deep lung penetration, strong link to mortality
- $\mathrm{PM10}$ = coarse fraction, still harmful
- $\mathrm{NO_2}$ = respiratory irritant, traffic-related
- $\mathrm{O_3}$ = oxidant, health effects at high levels
- $\mathrm{SO_2}$ = irritant, often lower in modern cities

Relative to EU/WHO annual limits, exceedances are often most frequent for PM2.5 or NO$_2$ in urban traffic sites.

The next step is to explore how PM2.5 concentrations vary across time and geographic locations.

For the presentation, include only:

- [images/pollutant_counts.csv](images/pollutant_counts.csv)
- one boxplot or histogram

# 4. Key Discoveries from EDA

Combine sections 4, 5, and 6 into three discoveries.

After identifying PM2.5 as the focus pollutant, we explored how pollution behaves across monitoring stations, time, and geographic locations.

The exploratory analysis reveals three key insights.

## Discovery 1 --- Pollution levels are highly uneven across monitoring stations

Next, we analyzed the distribution of pollution levels across observations.

### Supporting outputs

[images/pollution_level_distributions.png](images/pollution_level_distributions.png)

[images/overall_pollution_level_distributions.png](images/overall_pollution_level_distributions.png)

[images/secondary_pollution_level_distributions.png](images/secondary_pollution_level_distributions.png)

The distributions show that pollution values are strongly right-skewed. Most stations report moderate pollution levels, while a smaller number experience very high concentrations. These extreme values suggest that pollution exposure is not evenly distributed, and that certain areas may consistently experience worse air quality.

## Discovery 2 --- Pollution patterns change over time

Next, we analyzed temporal changes in pollution levels between 2015 and 2024.

### Supporting outputs

[images/temporal_trend_distributions.png](images/temporal_trend_distributions.png)

[images/YoY_change_in_median.png](images/YoY_change_in_median.png)

[images/YoY_secondary_pollutans_median.png](images/YoY_secondary_pollutans_median.png)

The analysis reveals noticeable year-to-year fluctuations in pollution levels. A particularly visible change occurs around 2020, when pollution levels decrease across several pollutants. This likely reflects the impact of COVID-19 lockdowns, which temporarily reduced traffic and industrial emissions. In subsequent years, pollution levels begin to increase again, suggesting that the drop was temporary rather than structural.

Pollution levels are dynamic and influenced by external events, indicating that environmental and socio-economic factors can significantly affect air quality.

## Discovery 3 --- Pollution varies strongly across locations

To understand spatial variation in pollution levels, we compared pollution concentrations across countries, cities, and monitoring station contexts.

### Supporting outputs

[images/average_pollution_by_country.png](images/average_pollution_by_country.png)

[images/most_polluted_cities.png](images/most_polluted_cities.png)

[images/station_context.png](images/station_context.png)

[images/secondary_by_station.png](images/secondary_by_station.png)

### Cross-country comparison

The first figure compares the median pollution levels by country for the main pollutants.

**Key observations:**

- Italy consistently shows the highest median values among the compared countries (Italy, Switzerland, Austria, France) for NO$_2$, PM10, and PM2.5.
- Switzerland generally exhibits lower median concentrations for particulate matter pollutants.
- O$_3$ levels are relatively similar across countries, with smaller differences compared to other pollutants.

This suggests that pollution levels may depend on national environmental conditions, urban density, and emission sources, motivating a deeper focus on Italy, where pollution levels appear relatively higher.

### City-level pollution hotspots

The second figure shows the top 15 most polluted cities per pollutant.

**Key observations:**

- Many of the highest pollution values occur in smaller municipalities rather than only large metropolitan areas.
- Cities such as Soresina, Revello, and Dalmine appear among the highest for PM2.5.
- For NO$_2$, larger urban areas such as Milano and Torino also appear among the most polluted locations.

Pollution hotspots are not limited to major cities and may depend on local industrial activity, geography, or traffic conditions.

### Station context: type and area

The third and fourth figures compare pollution levels across monitoring station types and areas.

**Key observations:**

**Station type**

- Traffic stations record the highest median pollution levels for NO$_2$, PM10, and PM2.5.
- Background stations show lower concentrations, representing general ambient conditions.
- Industrial stations show intermediate values.

This is expected because traffic stations are typically placed near major roads or high-emission areas.

**Station area**

Comparing rural, suburban, and urban areas:

- Urban stations tend to show higher NO$_2$ and PM concentrations, reflecting higher traffic density and urban emissions.
- Rural areas generally exhibit lower concentrations for most pollutants.
- O$_3$ shows the opposite pattern, with higher values in rural areas, which is consistent with known atmospheric chemistry effects.

### Secondary pollutants

The secondary pollutant comparison confirms similar spatial patterns:

- Higher concentrations of pollutants such as SO$_2$ and CO are associated with traffic or industrial station types.
- Urban areas tend to show higher concentrations compared to rural areas.

## Health-Weighted Pollution Exposure (ALISA'S HEATMAP)

While the previous analysis focuses on pollution concentrations themselves, an important question for policymakers is how these pollution patterns translate into potential health impacts.

To better illustrate this dimension, we constructed a Health-Weighted Pollution Index map, which combines pollutant concentrations with their relative health impact.

### Supporting output

[images/health_weighted_pollution_index_map.png](images/health_weighted_pollution_index_map.png)

This map highlights regions where pollution exposure may pose greater risks to human health, rather than simply showing areas with high pollutant concentrations.

In particular, the map reveals that several areas with elevated PM2.5 levels correspond to regions where population exposure may be especially concerning, reinforcing the relevance of focusing on PM2.5 in the subsequent modeling stage.

This visualization moves the analysis from pollution concentrations to potential human impact, which is why PM2.5 becomes the key pollutant for predictive modeling.

## Key Conclusion from the EDA

The exploratory analysis reveals several important patterns in the air quality dataset.

First, pollution levels are highly uneven across monitoring stations. While most locations exhibit moderate pollution levels, a smaller number of stations experience significantly higher concentrations. This indicates that pollution exposure is spatially concentrated rather than uniformly distributed.

Second, pollution levels vary over time. The temporal analysis shows noticeable fluctuations between years, including a temporary decrease around 2020 that likely reflects the impact of reduced mobility during COVID-19 lockdowns.

Third, pollution levels vary systematically across geographic contexts. Differences appear between countries, cities, monitoring station types, and urban versus rural areas. Traffic monitoring stations and urban environments generally show higher concentrations of pollutants such as NO$_2$ and PM2.5.

Finally, the Health-Weighted Pollution Index highlights areas where pollution exposure may have greater implications for human health, reinforcing the importance of focusing on PM2.5 as the primary pollutant in the subsequent analysis.

Taken together, these findings suggest that pollution patterns are influenced by a combination of environmental conditions, urban characteristics, and human activity. This motivates the next stage of the project, which introduces contextual environmental variables that may help explain these patterns and support predictive modeling.

# 5. Predictive Opportunities and What Drives Them

These variables are introduced to test whether environmental context can explain the spatial pollution differences identified in the exploratory analysis.

Our contextual variables provide a broader description of the environmental conditions that may influence pollution accumulation and dispersion. Understanding these relationships is essential for developing models capable of predicting pollution levels and assessing their potential societal impact.

## 5.1 Relationships Between Pollutants

Before introducing environmental drivers, we first explored relationships between pollutants already present in the dataset.

### Supporting outputs

[images/correlation_matrix.png](images/correlation_matrix.png)

[images/extended_correlation_matrix.png](images/extended_correlation_matrix.png)

[images/pairs_pollutants_scatterplot.png](images/pairs_pollutants_scatterplot.png)

The correlation matrices reveal several important relationships.

Pollutants such as PM2.5 and PM10 show strong positive correlations, indicating that they often originate from similar emission sources such as traffic and industrial activity.

Nitrogen dioxide (NO$_2$), which is closely linked to vehicle emissions, also exhibits positive correlations with particulate matter.

These relationships suggest that multiple pollutants may share common emission drivers, reinforcing the importance of understanding environmental and urban factors that influence pollution generation and dispersion.

## 5.2 Elevation and Pollution

One environmental factor that may influence pollution levels is elevation.

### Supporting outputs

[images/pm25_vs_elevation_regression.png](images/pm25_vs_elevation_regression.png)

[images/pm25_by_elevation_quartile.png](images/pm25_by_elevation_quartile.png)

The analysis shows a weak negative relationship between PM2.5 levels and elevation. Locations situated at higher altitudes tend to experience slightly lower PM2.5 concentrations, which may be explained by atmospheric dynamics.

The quartile analysis further confirms that lower elevation areas tend to exhibit higher PM2.5 concentrations on average.

## 5.3 Population Density and Urban Activity

Another important factor influencing pollution levels is population density, which serves as a proxy for urban activity.

Areas with higher population density typically experience:

- greater traffic intensity
- increased economic activity
- higher energy consumption

These factors can lead to higher emissions of pollutants such as NO$_2$ and particulate matter.

As discussed earlier in Section 2.6, the population information included in the original dataset remains constant across years. Therefore, we integrate updated population data from ISTAT, allowing future models to better capture the relationship between population distribution and pollution exposure.

## 5.4 Green Space and Pollution Mitigation

Urban green spaces may also influence air quality by helping to:

- absorb certain pollutants
- improve local air circulation
- mitigate urban heat effects

Cities with higher proportions of green space may therefore experience lower pollution concentrations compared to densely built urban environments. To explore this effect, we plan to incorporate green space ratios into the dataset, which will allow us to analyze how urban planning characteristics relate to pollution levels.

## 5.5 Potential Predictive Applications

Based on the exploratory analysis, three potential directions emerge.

### 1. Pollution forecasting

Using environmental and urban variables, it becomes possible to develop models that predict PM2.5 levels across locations and time. Such models could help identify future pollution hotspots and support environmental monitoring.

### 2. Public health risk prediction

PM2.5 is strongly associated with respiratory and cardiovascular diseases. Combining pollution predictions with demographic information (such as population density and age distribution) could help estimate:

- respiratory disease risk
- hospital admission surges
- healthcare demand

### 3. Urban planning and environmental policy support

Environmental variables such as green space availability, elevation, and population density can help identify urban areas that are more vulnerable to pollution accumulation.

Such insights can support:

- urban planning decisions
- pollution mitigation strategies
- environmental policy development

## Conclusion

Taken together, the analysis suggests that environmental context plays a significant role in shaping pollution patterns.

By combining pollution measurements with contextual variables such as elevation, population density, and green space availability, it becomes possible to move from descriptive analysis toward predictive modeling.

The next section therefore focuses on how these variables can be integrated into a modeling framework for predicting PM2.5 levels and their potential impacts.

# 6. Future Modeling Direction and Potential Impact

The exploratory analysis and environmental enrichment steps provide a foundation for developing predictive models of PM2.5 pollution levels. By combining pollution measurements with contextual variables, the project can move from descriptive analysis toward predictive and decision-support models. Several modeling approaches could be implemented.

## 6.1 PM2.5 Prediction Model

The first step is to build a model capable of predicting PM2.5 concentrations across locations and time.

Potential input variables include:

- other pollutant levels (PM10, NO$_2$, O$_3$)
- elevation
- population density
- urban characteristics such as green space ratios
- monitoring station context (traffic, background, industrial)

Possible modeling approaches:

- linear regression models
- tree-based models (Random Forest, Gradient Boosting)
- spatial regression methods

### Potential outputs

Such a model could produce:

- predicted PM2.5 levels for different cities
- identification of high-risk pollution hotspots
- estimated contribution of different environmental drivers

## 6.2 Health Risk Forecasting

PM2.5 pollution is strongly linked to respiratory and cardiovascular diseases. By combining PM2.5 predictions with demographic information, the model could be extended to estimate potential health impacts of pollution exposure.

Possible outputs include:

- predicted respiratory disease risk
- expected hospital admission surges
- estimated public health burden associated with pollution

These insights could help public health authorities anticipate healthcare demand and design targeted interventions.

## 6.3 Urban Policy and Environmental Planning

The modeling framework could also support policy and urban planning decisions. By analyzing how variables such as population density, green space availability, and elevation influence pollution levels, the model could identify:

- urban areas that are more vulnerable to pollution accumulation
- locations where increasing green spaces may reduce pollution exposure
- regions where emission reduction policies may have the greatest impact

### Potential impact

These insights could support:

- urban planning strategies
- environmental regulation
- air quality monitoring programs

By identifying the key drivers of PM2.5 pollution and forecasting pollution levels, the project can support data-driven decision making for policymakers, healthcare systems, and environmental agencies.