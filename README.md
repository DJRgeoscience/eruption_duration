# Survival analysis of volcanic eruption duration

Included are several scripts for python3 used to build time-to-event models for eruption duration.<br>

Input data were gathered from databases run by Smithsonian's Global Volcanism Program (GVP).<br>

Two types of eruptions are considered: pulses and events.<br>
Pulse - relatively short eruption (generally <1 week) that is considered to be continuous or semi-continuous.<br>
Event - generally longer eruption that can include breaks in activity of 90 days or less.

## Files

### Web scraping
scrape_gvp.py - Web scraping tool to get Weekly Reports from the GVP. Uses Selenium and Google Chrome.<br>

### Cox Proportional-Hazards models
event_duration_CPH.py<br>
pulse_duration_CPH.py<br>

### Cox Proportional-Hazards models with ridge penalty
event_duration_CPH-ridge.py<br>
pulse_duration_CPH-ridge.py<br>

### Cox Proportional-Hazards models with lasso penalty
event_duration_CPH-lasso.py<br>
pulse_duration_CPH-lasso.py<br>

### Cox Proportional-Hazards models with elastic net penalty
event_duration_CPH-elastic_net.py<br>
pulse_duration_CPH-elastic_net.py<br>

### Random Survival Forest models
event_duration_RSF.py<br>
pulse_duration_RSF.py<br>

### Survival analysis with neural networks and Cox regression (Cox-Time) - in progress
event_duration_CoxTime.py<br>
pulse_duration_CoxTime.py<br>