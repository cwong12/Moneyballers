#!/usr/bin/python

import sqlite3
import numpy as np
import matplotlib.pyplot as plt


#f = open("./gl2000-2009/")

connection = sqlite3.connect("data.db")
cursor = connection.cursor()
cursor.execute("SELECT b.home_line, b.away_line FROM betting_data as b WHERE b.home_team = 'STL' AND b.away_team = 'CHC';")
results = cursor.fetchall()

awayScore = []
count = 1
counts = []
homeScore = []
for r in results:
    print(count)
    counts.append(count)
    count+=1

    homeScore.append(r[0])
    awayScore.append(r[1])

    print(r)

plt.scatter(counts, homeScore, color="red")
plt.scatter(counts, awayScore, color="blue")
plt.legend(['Cards', 'Cubs'])
plt.xlabel('Game')
plt.ylabel('Scores')
plt.show();
cursor.close()
connection.close()
