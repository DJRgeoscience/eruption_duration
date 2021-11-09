# -*- coding: utf-8 -*-
            
#%% Update and initiate Chrome driver

from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver

driver = webdriver.Chrome(ChromeDriverManager().install())

#%% Scrape Weeklies
import csv
from selenium.webdriver.common.by import By
from datetime import date, datetime, timedelta

def find_reports(pull, out):
    for p in pull:
        t = p.get_attribute('id')
        t = t.split( sep='_' )
        temp = [ t[1] ]
        p = p.text
        if '|' in p:
            p = p.split( sep='|' )
            temp.append( p[0][:-1] )
            d = p[1].split( sep='-' )
            dd = d[0][1:] + ' ' + d[-1][-5:-1]
            datetime_object = datetime.strptime(dd, '%d %B %Y')
            temp.append( datetime_object.strftime('%m/%d/%Y') )
            out.append( temp )
    return out

scrape = []

s = '20200729' #start scraping the next week
url_base = 'https://volcano.si.edu/reports_weekly.cfm?weekstart='
delta = timedelta(days=7)

count = 0
while True:
    
    ymd = date( int(s[0:4]), int(s[4:6]), int(s[6:]) )
    ymd += delta
    s = ymd.strftime('%Y%m%d')
    print(s)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load website and display all information
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    url = url_base+s

    # Call website
    driver.get(url)
    
    # Allow 4 seconds of load time
    driver.implicitly_wait(4)
    
    # Pull weeklies
    out = [] #initialize output
    
    try:
        pull = driver.find_elements_by_xpath(".//tr[@class='WeeklyNameNew']")
        out = find_reports(pull, out)
    except:
        print('Error 1')
    
    try:
        for i in range(len(out)):
            weekly = driver.find_element(By.XPATH,'/html/body/div[3]/div[1]/div[1]/table[1]/tbody/tr[{}]/td[2]/p[1]'.format( (i+1)*2 )).text
            out[i].append( weekly )
    except:
        print('Error 2')
    
    len_new = len(out)
    
    try:
        pull = driver.find_elements_by_xpath(".//tr[@class='WeeklyNameContinuing']")
        out = find_reports(pull, out)
    except:
        print('Error 3')
    
    try:
        for i in range( len(out) - len_new ):
            weekly = driver.find_element(By.XPATH,'/html/body/div[3]/div[1]/div[1]/table[2]/tbody/tr[{}]/td[2]/p[1]'.format( (i+1)*2 )).text
            out[ len_new + i ].append( weekly )
    except:
        try:
            for i in range( len(out) - len_new ):
                weekly = driver.find_element(By.XPATH,'/html/body/div[3]/div[1]/div[1]/table/tbody/tr[{}]/td[2]/p[1]'.format( (i+1)*2 )).text
                out[ len_new + i ].append( weekly )
        except:
            print('Error 4')
    
    scrape += out
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Break out of loop, if necessary
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if s == '20210929': # Last Weekly captured, break loop
        break
    
# Output scraped results
with open('input/scraped_weeklies.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(scrape)
    