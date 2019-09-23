# MLB-Data

<html>
<body>
<p>(COMMIT, PULL, PUSH)
Here is our blog.</p>

<p>Blog Post 1:</p>

<p>Problem Statement (Question):</p>

<p>Can we use historical MLB game data to predict future outcomes of games?</p>

<p>Goals Coming in:
Comprehend 161 data types and download the 25 years of data sets off retro sheet.
Begin to clean dataset by removing all games where column 161 does not equal “Y”</p>

<p>Items Accomplished: </p>

<ol>
<li>Comprehended 161 data types and narrowed the data sets down to 18 years from 2000 to 2018 and narrowed down features to about 20-30 features from this data set alone. This data was acquired from retrosheet.com which stores 161 different attributes for each game such as the teams, score by inning, and player data. We elected not to collect much of the by-inning data because this information will not be available before a bet must be placed on the game, and we plan to use this model before each game, not during the game. There are 2,430 games played each season plus postseason games. We collected data for 18 years so our database contains around 45,000 games. </li>

<li>Incomplete data and odd data lines including games during protests/games that were rained out were removed. This was necessary to parse the data correctly, and these games represent a very small amount of games, so that we are not removing a significant number of games. We ran into one edgecase where no a baseball game had no attendees. The attendance was stored as null which caused us problems in adding it to SQL so we turned this into a zero so that the game could be added. However, a game is extremely anomalous(only one ever) so we will most likely remove it in the future. </li>

<li>Unique IDs were then generated for each of the games in the format date-hometeam-visitingteam-numberofteam. We need a unique ID for each game because there is no single attribute that is unique to each game. We can use this unique ID to pair the game data with the betting data since they come from different sources and must be linked.</li>

<li>Inserted these data points into an sql database. We also ported it into a pandas dataframe. These will allow for easy viewing and processing of the data in the future.</li>
</ol>

<p><img src="https://github.com/jakeacquadro/MLB-Data/blob/master/sampledata.png" alt="Sample of some records in our database" /></p>

<p>Future Direction:</p>

<ol>
<li>Acquire scraped betting data from oddsportal.com. We plan to use BeautifulSoup to scrape the betting odds and information about these games so we can pair them with the rest of the data about the games.</li>

<li>Finish cleaning the data we have scraped, looking for incomplete entries. We don't want to consider games that are missing attributes.</li>

<li>Join scraped betting data with game data in sql database. The betting data will become an attribute along with other attributes such as home team, umpire, and starting pitcher.</li>

<li>Perform logistical regression and find odd ratio comparison for features and odd ratio comparison for environment vs. player statistics for one year.</li>

<li>Research several different models we can use besides logistical regressions to investigate data such as ours with a binary output (win or loss).</li>
</ol>game data to predict future outcomes of games?

Goals Coming in:
Comprehend 161 data types and download the 25 years of data sets off retro sheet.
Begin to clean dataset by removing all games where column 161 does not equal “Y”

Items Accomplished: 
1. Comprehended 161 data types and narrowed the data sets down to 18 years from 2000 to 2018 and narrowed down features to about 20-30 features from this data set alone. This data was acquired from retrosheet.com which stores 161 different attributes for each game such as the teams, score by inning, and player data. We elected not to collect much of the by-inning data because this information will not be available before a bet must be placed on the game, and we plan to use this model before each game, not during the game. There are 2,430 games played each season plus postseason games. We collected data for 18 years so our database contains around 45,000 games. 
2. Incomplete data and odd data lines including games during protests/games that were rained out were removed. This was necessary to parse the data correctly, and these games represent a very small amount of games, so that we are not removing a significant number of games. We ran into one edgecase where no a baseball game had no attendees. The attendance was stored as null which caused us problems in adding it to SQL so we turned this into a zero so that the game could be added. However, a game is extremely anomalous(only one ever) so we will most likely remove it in the future. 
3. Unique IDs were then generated for each of the games in the format date-hometeam-visitingteam-numberofteam. We need a unique ID for each game because there is no single attribute that is unique to each game. We can use this unique ID to pair the game data with the betting data since they come from different sources and must be linked.
4. Inserted these data points into an sql database. We also ported it into a pandas dataframe. These will allow for easy viewing and processing of the data in the future.

![Sample of some records in our database](https://github.com/jakeacquadro/MLB-Data/blob/master/sampledata.png)

Future Direction:

1. Acquire scraped betting data from oddsportal.com. We plan to use BeautifulSoup to scrape the betting odds and information about these games so we can pair them with the rest of the data about the games.
2. Finish cleaning the data we have scraped, looking for incomplete entries. We don't want to consider games that are missing attributes.
3. Join scraped betting data with game data in sql database. The betting data will become an attribute along with other attributes such as home team, umpire, and starting pitcher.
4. Perform logistical regression and find odd ratio comparison for features and odd ratio comparison for environment vs. player statistics for one year.
5. Research several different models we can use besides logistical regressions to investigate data such as ours with a binary output (win or loss).
</body>
<script>
const data = unpickle("forest_model.joblib")
console.log(data)
</script>
</html>
