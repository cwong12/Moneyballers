import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import statsmodels.api as sm
from patsy import dmatrices
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from joblib import dump, load
from sklearn_porter import Porter


def teamToStadiumMap(gamedf):

    homeTeams = gamedf.home_team.unique()
    homeTeams.sort()
    # print(homeTeams)

    stadDict = {}

    with open('MLB_Stadiums.csv', 'r') as csvfile:
        stadreader = csv.reader(csvfile)

        for row in stadreader:
            coords = [float(row[3]), float(row[4])]
            stadDict[row[0]] = coords
    return stadDict

def teamDist(gamedf):
    teamDict = teamToStadiumMap(gamedf)

    distList = []
    for row in gamedf.iterrows():
        home = row[1]["home_team"]
        away = row[1]["away_team"]



        homeCoords = np.array(teamDict[home])
        awayCoords = np.array(teamDict[away])


        dist = np.linalg.norm(homeCoords-awayCoords)
        distList.append(dist)
        # testgamedf["winning_team"] = testgamedf.apply(lambda row: row.home_team if row.game_winner == 1 else row.away_team, axis = 1)
    gamedf['distance_from_away_team'] = distList

def MultipleLinearRegression(X, y, linear_model):

    lm = linear_model

    params = np.append(lm.intercept_,lm.coef_)
    predictions = lm.predict(X)

    newX = np.append(np.ones((len(X),1)), X, axis=1)
    MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

    # var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
    # sd_b = np.sqrt(var_b)
    # ts_b = params/ sd_b
    odd_ratio = np.exp(params)

    # p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]

    myDF3 = pd.DataFrame()
    myDF3["Coefficients"],myDF3["Odd Ratio"] = [params,odd_ratio]
    print(myDF3)

def fullLogisticRegression(X,y, LogisticRegression):
    pass

def home_team_advantage_heatmap(gamedf):
    teamwin_data = gamedf[["home_team" ,"away_team","game_winner"]]
    team_percent_wins = teamwin_data.groupby(["home_team", "away_team"]).mean().unstack()
    # print(teamwin_data[(teamwin_data["home_team"]=="ANA") & (teamwin_data["visiting_team"]=="ARI")])
    plt.figure(figsize=(18,18))

    plt.title("Home Team's percent wins against Away teams")
    heatmap = sns.heatmap(team_percent_wins['game_winner'],cmap="YlOrRd", xticklabels=True,yticklabels=True, annot=True, cbar_kws={'label': 'Percent wins as a decimal'})
    plt.yticks(rotation=0)
    plt.show()

def home_away_line_scatter(gamedf):
    linewin_data = gamedf[["home_line" ,"away_line","game_winner"]]

    # print(teamwin_data[(teamwin_data["home_team"]=="ANA") & (teamwin_data["visiting_team"]=="ARI")])
    relscatter = sns.relplot(x="home_line", y="away_line",hue="game_winner", palette="pastel", alpha=0.5, data=linewin_data, height=7)
    relscatter.set(xlabel=' Home Line', ylabel='Away Line', title='Home vs. Away Line Effect on Winners')
    relscatter._legend.texts[0].set_text("")
    relscatter._legend.texts[1].set_text("Away Team")
    relscatter._legend.texts[2].set_text("Home Team")
    relscatter._legend.set_title("Game Winner")
    plt.show()

def prob_to_money_line(p):
    dec = 1/p
    if dec < 2:
        line = -100 / (dec - 1)
    else:
        line = (dec - 1) * 100
    return line

def show_forest(model, features):


    # Extract single tree
    estimator = model.estimators_[5]
    dot_data = StringIO()
    # Export as dot file
    export_graphviz(estimator, out_file="tree.dot",
                    class_names=True,
                    feature_names=features,
                    rounded=True, proportion=False,
                    precision=2, filled=True)

    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())




def randomForest(X_train,Y_train, X_test, Y_test):


    forest = RandomForestClassifier(
        max_depth=10, max_features=X_train.shape[1] // 3, min_samples_leaf=X_train.shape[0] // 20,
        n_estimators=1000, n_jobs=7, oob_score=False)
    forest.fit(X_train, Y_train)


    porter = Porter(forest, language='js')
    output = porter.export(embed_data=True)
    with open('RandomForestClassifier.js', 'w+') as f:
        f.write(output)
    show_forest(forest, list(X_train.columns))




    predictions = pd.DataFrame(

        forest.predict_proba(X_test),
        X_test.index, ["away","home"]
    )

    predictions['fcst']= predictions.apply(lambda df: 0 if df.idxmax(axis=1) == 'away' else 1,axis=1)

    return predictions





def line_to_payout(row,opposite=False, winner=False):
    if winner:
        look_at = row['pred_winner']
    else:
        look_at = row['team_to_bet_on']

    if opposite:
        if look_at == 1:
            line = row['away_line']
            pred_line = row['n_away_line']
        else:
            line = row['home_line']
            pred_line = row['n_home_line']

    else:
        if look_at == 1:
            line = row['home_line']
            pred_line = row['n_home_line']

        else:
            line = row['away_line']
            pred_line = row['n_away_line']
    if line < 0:
        line *= -1
        return 100 / line * row['bet']
    else:
        return row['bet'] * line / 100



def make_money(predictions, x,y):


    away_line = []
    home_line = []
    for prediction in predictions.iterrows():
        away_line.append(prob_to_money_line(prediction[1]['away']))
        home_line.append(prob_to_money_line(prediction[1]['home']))
    df = pd.DataFrame(x)
    df['game_winner'] = y
    df['n_away_line'] = away_line
    df['n_home_line'] = home_line

    df['team_to_bet_on'] = df.apply(lambda row: 1 if row['n_home_line'] - row['home_line'] <= 0 else 0, axis=1)
    df['pred_winner'] = predictions['fcst']
    df['bet'] = 100


    df['stand_to_gain'] = df.apply(line_to_payout, axis=1)
    # df['stand_to_gain_opposite'] = df.apply(line_to_payout,opposite=True, axis=1)

    df['stand_to_gain_winner'] = df.apply(line_to_payout,winner=True, axis=1)

    df['result'] = df.apply(
        lambda row: row['stand_to_gain'] if row['team_to_bet_on'] == row['game_winner'] else -row['bet'], axis = 1)
    # df['opposite_result'] = df.apply(
    #     lambda row: row['stand_to_gain_opposite'] if row['team_to_bet_on'] != row['game_winner'] else -row['bet'], axis = 1)
    df['winner_result'] = df.apply(
        lambda row: row['stand_to_gain_winner'] if row['pred_winner'] == row['game_winner'] else -row['bet'], axis = 1)

    print("Money from following strategy: ${}".format(sum(df['result'])))
     # print("Money from betting against strategy: ${}".format(sum(df['opposite_result'])))
    print("Money from betting on the predicted winner: ${}".format(sum(df['winner_result'])))

    return df[["result", "winner_result"]]



def logReg(y, X):
    mod = sm.Logit(y, X)
    res = mod.fit(skip_hessian=1)
    return mod, res

if __name__=='__main__':
    conn = sqlite3.connect('data.db')
    c = conn.cursor()

    allgamedf = pd.read_sql_query("Select date, number_of_game, day_of_week, home_team, visiting_team as away_team, day_night, park_ID, game_winner from games", conn)
    # allgamedf.info()

    dict = teamToStadiumMap(allgamedf)
    teamDist(allgamedf)
    #print(allgamedf)
    traingamedf = pd.read_sql_query("Select date, number_of_game, day_of_week, home_team, visiting_team as away_team, day_night, park_ID, game_winner from games where date < date('2018-01-01');", conn)
    testgamedf = pd.read_sql_query("select date, number_of_game, day_of_week, home_team, visiting_team as away_team, day_night, park_ID, game_winner from games where date >= date('2018-01-01');", conn)

    traingamedf["game_winner"] = traingamedf["game_winner"].astype(int)
    traingamedf["winning_team"] = traingamedf.apply(lambda row: row.home_team if row.game_winner == 1 else row.away_team, axis = 1)
    teamDist(traingamedf)

    testgamedf["game_winner"] = testgamedf["game_winner"].astype(int)
    testgamedf["winning_team"] = testgamedf.apply(lambda row: row.home_team if row.game_winner == 1 else row.away_team, axis = 1)
    teamDist(testgamedf)

    # allgamedf.info()
    allgamedf["game_winner"] = allgamedf["game_winner"].astype(int)
    allgamedf["winning_team"] = allgamedf.apply(lambda row: row.home_team if row.game_winner == 1 else row.away_team, axis = 1)


    all_data = pd.merge(pd.read_sql("select * from betting_data where (away_line > 100 or away_line < -100) and (home_line >100 or home_line <-100) and (home_line != -700)", conn), allgamedf , how="inner")
    all_data['favorite_is_home'] = np.sign(all_data['home_line']) == -1

    train_data = pd.merge(pd.read_sql("select * from betting_data", conn), traingamedf , how="inner" )
    train_data['favorite_is_home'] = np.sign(train_data['home_line']) == -1

    test_data = pd.merge(pd.read_sql("select * from betting_data", conn), testgamedf , how="inner")
    test_data['favorite_is_home'] = np.sign(test_data['home_line']) == -1
    # HEATMAP VISUALIZATION
    home_team_advantage_heatmap(allgamedf)
    home_away_line_scatter(test_data)

    #CSV USE
    # allgamecsvdf = pd.read_csv('betting.csv')
    # allgamecsvdf["game_winner"] = allgamecsvdf["game_winner"].astype(int)
    # allgamecsvdf["winning_team"] = allgamecsvdf.apply(lambda row: row.home_team if row.game_winner == 1 else row.visiting_team, axis = 1)
    print("\n ------------------------------RANDOM DATA SPLIT ONLY------------------------------------------")
    y,X = dmatrices("game_winner ~ number_of_game + day_of_week + home_team + away_team + day_night + distance_from_away_team + away_line + home_line", data=all_data, return_type='dataframe')

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

    mod, res = logReg(y_train,X_train)
    predictions = np.array(res.predict(X_test).tolist())
    prediction_labels = np.where(predictions > 0.5, 1, 0)
    accuracy = 1.0 - np.sum(((np.subtract(prediction_labels, y_test['game_winner']))**2))/len(y_test)
    print(res.summary())

    odd_ratio = np.exp(res.params)
    for line in odd_ratio:
        print(line)

    print("Accuracy:", accuracy)
    print("\n ------------------------------VALIDATION WITH 2018 ONLY------------------------------------------")
    train_y,train_X = dmatrices("game_winner ~ number_of_game + day_of_week + home_team + away_team + day_night + distance_from_away_team + away_line + home_line", data=train_data, return_type='dataframe')

    y_2018,X_2018 = dmatrices("game_winner ~ number_of_game + day_of_week + home_team + away_team + day_night + distance_from_away_team + away_line + home_line", data=test_data, return_type='dataframe')

    print(len(y_2018))
    mod2, res2 = logReg(train_y,train_X)
    predictions2 = np.array(res2.predict(X_2018).tolist())

    pred_for = randomForest(train_X,train_y,X_2018,y_2018)

    print("RANDOM FOREST")
    forest_money = make_money(pred_for,X_2018, y_2018)
    forest_money['total_spent'] = np.arange(forest_money.shape[0]*100,step=100)
    print("LOGISTIC")
    pred3 = pd.DataFrame()
    pred3['home'] = predictions2
    pred3['away'] = 1-pred3['home']
    pred3['fcst'] = pred3.apply(lambda row: 0 if row.idxmax(axis=1) == 'away' else 1,axis=1)
    log_money = make_money(pred3, X_2018, y_2018)


    all_money = {}
    all_money["forest"] = {'data': forest_money.cumsum(axis=0), "color": 'green', 'ROI': forest_money.cumsum(axis=0)}
    all_money["forest"]
    all_money["logistic"] = {'data': log_money.cumsum(axis=0), "color": 'red'}

    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212,sharex=ax1)

    for model in all_money:
        ax1.plot('result', data=all_money[model]['data'], marker='', color=all_money[model]['color'], linewidth=2)
        ax1.set_ylabel("$ in profit")
        ax1.set_title("Money Earned from Betting on the Favorable Odds ")

        ax2.plot('winner_result', data=all_money[model]['data'], marker='', color=all_money[model]['color'],
                 linewidth=2)
        ax2.set_ylabel("$ in profit")
        ax2.set_title("Money Earned from Betting on Predicted Winner")

    plt.xlabel("Games for the Predicted 2018 Season")
    plt.ylabel("$ in profit")
    plt.legend(('Random Forest Classifier', 'Logistic Regression'), loc=0)
    plt.show()





    prediction_labels2 = np.where(predictions2 > 0.5, 1, 0)
    accuracy2 = 1.0 - np.sum(((np.subtract(prediction_labels2, y_2018['game_winner']))**2))/len(y_test)
    print(res2.summary())

    odd_ratio2 = np.exp(res2.params)
    for line in odd_ratio2:
        print(line)

    print("Accuracy:", accuracy2)
