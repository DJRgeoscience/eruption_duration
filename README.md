# Survival analysis of volcanic durations

Included are several scripts for python3 used to build time-to-event models that give probabalistic estimates of the duration of eruptive activity at volcanoes globally.<br>

Input data were gathered from databases run by Smithsonian's Global Volcanism Program (GVP).<br>

Two types of eruptions are considered: events and eruptions.<br>
Event - relatively short volcanic activity (generally <1 week) that is considered to be continuous or semi-continuous.<br>
Eruption - eruptive activity that ends with a significant (>90 days) break in activity.<br>

## Files

### Web scraping
scrape_gvp.py - Web scraping tool to get Weekly Reports from the GVP. Uses Selenium and Google Chrome.<br>

### Cox Proportional-Hazards models
event_duration_CPH.py<br>
eruption_duration_CPH.py<br>

### Cox Proportional-Hazards models with ridge penalty
event_duration_CPH-ridge.py<br>
eruption_duration_CPH-ridge.py<br>

### Cox Proportional-Hazards models with lasso penalty
event_duration_CPH-lasso.py<br>
eruption_duration_CPH-lasso.py<br>

### Cox Proportional-Hazards models with elastic net penalty
event_duration_CPH-elastic_net.py<br>
eruption_duration_CPH-elastic_net.py<br>

### Random Survival Forest models
event_duration_RSF.py<br>
eruption_duration_RSF.py<br>

### Gradient boosted models using Cox's partial likelihood
event_duration_GB.py<br>
eruption_duration_GB.py<br>

### Componentwise gradient boosted models using Cox's partial likelihood - in progress
event_duration_CGB.py<br>
eruption_duration_CGB.py<br>

### XGBoost models - in progress
event_duration_XGB.py<br>
eruption_duration_XGB.py<br>

### Survival analysis with neural networks and Cox regression (Cox-Time) - in progress
event_duration_CoxTime.py<br>
eruption_duration_CoxTime.py<br>