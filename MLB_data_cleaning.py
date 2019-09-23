from bs4 import BeautifulSoup
import requests
import sqlite3
import json
import zipfile, urllib, os
from urllib.request import Request,urlopen, urlretrieve
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas
import binascii

# betting data website: https://www.indatabet.com/baseball.html


def uniqueIDGen(date, ht, vt, num_g):
    unique_string = date[1:-1]+ht[1:-1]+vt[1:-1]+num_g[1:-1]
    return unique_string




season_data_2000_2009_url = 'https://www.retrosheet.org/gamelogs/gl2000_09.zip'
season_data_2010_2018_url = 'https://www.retrosheet.org/gamelogs/gl2010_18.zip'

season_data_2018_url = 'https://www.retrosheet.org/gamelogs/gl2018.zip'

req = urllib.request.Request(season_data_2018_url)
response = urllib.request.urlopen(req)
page = response.read()

# Create connection to database
# automatically creates new .db file if it doesn't exist
conn = sqlite3.connect('data.db')
c = conn.cursor()

# Delete tables if they exist
c.execute('DROP TABLE IF EXISTS "games";')

# Create table NEED TO PUT IN EACH COLUMN AND DATA TYPE: CONVERT
# TO THESE DATA TYPES IN FOR LOOP



create_symbols_table = """ CREATE TABLE games (
                            id text primary key,
                            date date,
                            number_of_game real,
                            day_of_week text,
                            visiting_team text,
                            visiting_team_league text,
                            visiting_team_game_number real,
                            home_team text,
                            home_team_league text,
                            home_team_game_number real,
                            visiting_team_score real,
                            home_team_score real,
                            length_of_game real,
                            day_night real,
                            park_ID real,
                            attendance real,
                            time_of_game real,
                            visiting_line_score text,
                            home_line_score text,
                            home_plate_umpire_ID text,
                            home_plate_umpire_name text,
                            visiting_team_manager_ID text,
                            visiting_team_manager_name text,
                            home_team_manager_ID text,
                            home_team_manager_name text,
                            visiting_starting_pitcher_ID text,
                            visiting_starting_pitcher_name text,
                            home_starting_pitcher_ID text,
                            home_starting_pitcher_name text,
                            home_player_1_id text,
                            home_player_1_name text,
                            home_player_1_position real,
                            home_player_2_id text,
                            home_player_2_name text,
                            home_player_2_position real,
                            home_player_3_id text,
                            home_player_3_name text,
                            home_player_3_position real,
                            home_player_4_id text,
                            home_player_4_name text,
                            home_player_4_position real,
                            home_player_5_id text,
                            home_player_5_name text,
                            home_player_5_position real,
                            home_player_6_id text,
                            home_player_6_name text,
                            home_player_6_position real,
                            home_player_7_id text,
                            home_player_7_name text,
                            home_player_7_position real,
                            home_player_8_id text,
                            home_player_8_name text,
                            home_player_8_position real,
                            home_player_9_id text,
                            home_player_9_name text,
                            home_player_9_position real,
                            visitng_player_1_id text,
                            visitng_player_1_name text,
                            visitng_player_1_position real,
                            visitng_player_2_id text,
                            visitng_player_2_name text,
                            visitng_player_2_position real,
                            visitng_player_3_id text,
                            visitng_player_3_name text,
                            visitng_player_3_position real,
                            visitng_player_4_id text,
                            visitng_player_4_name text,
                            visitng_player_4_position real,
                            visitng_player_5_id text,
                            visitng_player_5_name text,
                            visitng_player_5_position real,
                            visitng_player_6_id text,
                            visitng_player_6_name text,
                            visitng_player_6_position real,
                            visitng_player_7_id text,
                            visitng_player_7_name text,
                            visitng_player_7_position real,
                            visitng_player_8_id text,
                            visitng_player_8_name text,
                            visitng_player_8_position real,
                            visitng_player_9_id text,
                            visitng_player_9_name text,
                            visitng_player_9_position real,



                            acquisition_information text,
                            game_winner real
                            ); """
c.execute(create_symbols_table)
conn.commit()

#Test for 2018 url
req2 = urlopen(season_data_2018_url)
myZip2 = zipfile.ZipFile(BytesIO(req2.read()))
lines = myZip2.open('GL2018.TXT').readlines()

games_list = []

data_2000_2009 = urlopen(season_data_2000_2009_url)
zip_2000_2009 = zipfile.ZipFile(BytesIO(data_2000_2009.read()))
file_list_2000_2009 = zip_2000_2009.namelist()
seasons_2000_2009 = []
for season in file_list_2000_2009:
    current_season = zip_2000_2009.open(season).readlines()
    seasons_2000_2009.append(current_season)

data_2010_2018 = urlopen(season_data_2010_2018_url)
zip_2010_2018 = zipfile.ZipFile(BytesIO(data_2010_2018.read()))
file_list_2010_2018 = zip_2010_2018.namelist()
seasons_2010_2018 = []
for season in file_list_2010_2018:
    current_season = zip_2010_2018.open(season).readlines()
    seasons_2010_2018.append(current_season)

# split seasons into games, creating list of games
games_2000_2009 = []
for season in seasons_2000_2009:
    for game in season:
        game  = game.decode('utf-8')
        game = game.split(',')
        if len(game) == 161:
            if game[17] != "":
                #9 is away 10 is home
                if(game[2] == '"Mon"'):
                    game[2] = 1
                if(game[2] == '"Tue"'):
                    game[2] = 2
                if(game[2] == '"Wed"'):
                    game[2] = 3
                if(game[2] == '"Thu"'):
                    game[2] = 4
                if(game[2] == '"Fri"'):
                    game[2] = 5
                if(game[2] == '"Sat"'):
                    game[2] = 6
                if(game[2] == '"Sun"'):
                    game[2] = 7
                if(game[3] == '"MON"'):
                    continue
                if(game[6] == '"MON"'):
                    continue
                if(game[3] == '"FLO"'):
                    game[3] = '"MIA"'
                if(game[6] == '"FLO"'):
                    game[6] = '"MIA"'
                if(game[12] == '"D"'):
                    game[12] = 1
                if(game[12] == '"N"'):
                    game[12] = 0
                if(game[9] > game[10]):
                    game.append(0)
                if(game[10] > game[9]):
                    game.append(1)
                if len(game) == 162:
                    games_2000_2009.append(game)

games_2010_2018 = []
for season in seasons_2010_2018:
    for game in season:
        game = game.decode('utf-8')
        game = game.split(',')
        if len(game) == 161:
            if game[17] != "":
                #9 is away 10 is home
                if(game[2] == '"Mon"'):
                    game[2] = 1
                if(game[2] == '"Tue"'):
                    game[2] = 2
                if(game[2] == '"Wed"'):
                    game[2] = 3
                if(game[2] == '"Thu"'):
                    game[2] = 4
                if(game[2] == '"Fri"'):
                    game[2] = 5
                if(game[2] == '"Sat"'):
                    game[2] = 6
                if(game[2] == '"Sun"'):
                    game[2] = 7
                if(game[3] == '"MON"'):
                    continue
                if(game[6] == '"MON"'):
                    continue
                if(game[3] == '"FLO"'):
                    game[3] = '"MIA"'
                if(game[6] == '"FLO"'):
                    game[6] = '"MIA"'
                if(game[12] == '"D"'):
                    game[12] = 1
                if(game[12] == '"N"'):
                    game[12] = 0
                if(game[9] > game[10]):
                    game.append(0)
                if(game[10] > game[9]):
                    game.append(1)
                if len(game) == 162:
                    games_2010_2018.append(game)
            else:
                print("incomplete")
                print(game[100])


# for i in range(len(games_2000_2009)):
#     games_2000_2009[i][0] = games_2000_2009[i][0].split()
#
# for i in range(len(games_2010_2018)):
#     games_2010_2018[i][0] =
data_point_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,16,17,18,19,20,77,78,89,90,91,92,101,102,103,104]
for i in range(105,159):
    data_point_indices.append(i)
data_point_indices.append(160)
data_point_indices.append(161)

print(data_point_indices)

ranges = [games_2000_2009, games_2010_2018]
for season_range in ranges:
#for i in range(len(games_2000_2009)):
#    str_to_execute = 'INSERT INTO games VALUES ' + '('
#    for index in data_point_indices[:-1]:
#        str_to_execute += '{}, '.format(games_2000_2009[i][index])
#    str_to_execute += '{})'.format(games_2000_2009[i][-1])
#    c.execute(str_to_execute)

    for i in range(len(season_range)):
        unique_id = uniqueIDGen(season_range[i][0],season_range[i][6],season_range[i][3],season_range[i][1] )

        str_to_execute = 'INSERT INTO games VALUES ' + '("{}",'.format(unique_id)

        season_range[i][0] = '"{}-{}-{}"'.format(season_range[i][0][1:5],season_range[i][0][5:7],season_range[i][0][7:9])
        for index in data_point_indices[:-1]:
            # The game in baltimore where literally noone attended was recorded as None
            if season_range[i][14] != None:
                str_to_execute += '{}, '.format(season_range[i][index])
            else:
                str_to_execute += '0, '
        str_to_execute += '{})'.format(season_range[i][-1])
    #    print(str_to_execute)
        c.execute(str_to_execute)

conn.commit()



#edit next time to give each game a unique id

#ok
