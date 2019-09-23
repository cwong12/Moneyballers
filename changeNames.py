import sqlite3

if __name__=='__main__':
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    change_names = "UPDATE games SET home_team = 'LAD' WHERE home_team = 'LAA' AND home_team_league = 'NL'; "
    #change_names = "UPDATE games SET home_team = 'WSH' WHERE home_team = 'WAS'; "
    #change_names = "UPDATE games SET visiting_team = 'CWS' WHERE visiting_team = 'CHA'; "
    change_names = "SELECT count(distinct(home_team)) from games"
    #change_names = "UPDATE games SET home_team = 'CWS' WHERE home_team = 'CHA'; "
    #change_names = "UPDATE games SET visiting_team = 'SD' WHERE visiting_team = 'SDN'; "
    #change_names = "UPDATE games SET home_team = 'TB' WHERE home_team = 'TBA'; "
    c.execute(change_names)
    print(c.fetchall())
    conn.commit()
