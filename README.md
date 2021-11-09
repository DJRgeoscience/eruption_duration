# Survival analysis of volcanic eruption duration

Included are several scripts for python3 used to build time-to-event models for eruption duration.<br>

Input data were gathered from databases run by Smithsonian's Global Volcanism Program (GVP).<br>

Two types of eruptions are considered: pulses and events.<br>
Pulse - relatively short eruption (generally <1 week) that is considered to be continuous or semi-continuous.<br>
Event - generally longer eruption that can include breaks in activity of 90 days or less.

Files:<br>
event_duration_RSF.py - Random survival forest model for event durations.<br>

pulse_duration_RSF.py - Random survival forest model for pulse durations.<br>

scrape_gvp.py - Web scraping tool to get Weekly Reports from the GVP. Uses Selenium and Google Chrome.<br>
