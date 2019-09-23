#!/usr/bin/env python
# coding: utf-8

# In[155]:


from bs4 import BeautifulSoup
from selenium import webdriver
from bs4 import BeautifulSoup


# In[162]:


abbrvs = {"Arizona Diamondbacks":"ARI","Atlanta Braves":"ATL","Baltimore Orioles":"BAL","Boston Red Sox":"BOS","Chicago Cubs":"CHC","Chicago White Sox":"CHW","Cincinnati Reds":"CIN","Cleveland Indians":"CLE","Colorado Rockies":"COL","Detroit Tigers":"DET","Florida Marlins":"FLA","Houston Astros":"HOU","Kansas City Royals":"KAN","Los Angeles Angels":"LAA","Los Angeles Dodgers":"LAD","Milwaukee Brewers":"MIL","Minnesota Twins":"MIN","New York Mets":"NYM","New York Yankees":"NYY","Oakland Athletics":"OAK","Philadelphia Phillies":"PHI","Pittsburgh Pirates":"PIT","San Diego Padres":"SD","San Francisco Giants":"SF","Seattle Mariners":"SEA","St.Louis Cardinals":"STL","Tampa Bay Rays":"TB","Texas Rangers":"TEX","Toronto Blue Jays":"TOR","Washington Nationals":"WAS"}
browser = webdriver.Chrome('/Users/jakeacquadro/downloads/chromedriver')

return_list = []

# pages stores how many pages of info there are for each year
pages = [21, 17, 48, 59, 56, 57, 57, 58, 57, 57, 57, 58, 58]

for year in range(6, 19):
    year_index = year - 6
    page_count = pages[year_index] + 1
    
    # yearadd is what is inserted into the url
    yearadd = str(year)
    if year < 10:
        yearadd = '0' + str(year)
    base_URL = 'https://www.oddsportal.com/baseball/usa/mlb-20' + yearadd + '/results/#/'
    
    for z in range(1, page_count):
        URL = base_URL + 'page/' + str(z) + '/'
        browser.get(URL)
        soup = BeautifulSoup(browser.page_source, "lxml")
        #soup.prettify()
        table_info = soup.find('table', {"class": "table-main", 'id': 'tournamentTable'})

        # get the day each game occurred on
        days = []
        temp_days = table_info.find_all_next('span', {'class': lambda L: L and L.startswith('datet')})
        for day in temp_days:
            days.append(day.text)
        #print(days)

        # get the time of each game
        times = []
        temp_times = table_info.find_all_next('td', {'class': lambda K: K and K.startswith('table-time datet')})
        for time in temp_times:
            times.append(time.text)
        print(times)

        # get the teams in each game. Format: "team1 - team2"
        names = []
        temp_names = table_info.find_all_next('td', {'class': 'name table-participant'})
        for name in temp_names:
            bothnames = name.text.replace(u'\xa0', u' ')
            bothnames = bothnames.split('-')
            bothnames = [x.strip() for x in bothnames]
            #print(bothnames)
            team1 = bothnames[0]
            team2 = bothnames[1]
            team1abbrv = abbrvs[team1]
            team2abbrv = abbrvs[team2]
            names.append((team1abbrv, team2abbrv))
        print(names)

        # get the odds placed on team 1
        team1odds = []
        team2odds = []
        temp_odds1 = table_info.find_all_next('td', {'class': 'odds-nowrp'})
        temp_counter = 1
        for odd in temp_odds1:
            temp_counter += 1
            if temp_counter%2 == 0:
                team1odds.append(odd.text)
            else:
                team2odds.append(odd.text)
        #print(team1odds)
        #print(team2odds)


        # now we must count how many games occurred on each day so we can assign the right dates to the right games. This info was not stored together within the table
        counters = []
        counter = 0
        rows = list(table_info.find_all_next('tr'))

        for i in range(len(rows)):
            temp_class = rows[i].get('class')
            if temp_class == ['center', 'nob-border']:
                if counter > 0:
                    counters.append(counter)
                    counter = 0
            if temp_class == ['odd', 'deactivate'] or temp_class == ['', 'deactivate']:
                counter += 1
        counters.append(counter)
        #print(counters)

        # create dates list
        new_date_list = []
        for x in range(len(days)):
            num_games = counters[x]
            for i in range(num_games):
                new_date_list.append(days[x])
        #print(new_date_list)

        # assemble information
        for i in range(len(new_date_list)):
            unique_game = []
            unique_key = new_date_list[i] + times[i] + names[i]
            my_list = [unique_key, team1odds[i], team2odds[i]]
            return_list.append(my_list)
print(return_list)


# In[ ]:





# In[ ]:




