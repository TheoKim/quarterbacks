#%% 0) Init - import packages, set up data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier 

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

qb_data = pd.read_csv('QBStatsComposite6.csv')

# Setting where a QB has to start at least 8 games.
qb_data = qb_data[qb_data["GS"] >= 8]

# QB stats
x_qbr = qb_data['QBR']
x_pass_rating = qb_data['Passer_Rating']
x_TDs_p = qb_data['TD_Percentage']
x_Int_p = qb_data['Int_Percentage']
x_Yds_p_a = qb_data['Yards_per_attempt']
x_Adj_Yds_p_a = qb_data['Adjusted_yards_per_attempt']
x_n_Yds_p_att = qb_data['Net_yards_per_attempt']
x_Adj_n_Yds_p_att = qb_data['Adjusted_net_yards_per_attempt']
ranking = qb_data["Rank"]

y = qb_data['Win_Percentage']


#%% 0.5 Visualizing data


QB_metrics = [

    'QBR', 'Passer_Rating', 'Cmp_Percentage', 'Yds',
    'TD', 'TD_Percentage', 'Int', 'Int_Percentage', 'Yards_per_attempt',
    'Adjusted_yards_per_attempt', 'Yards_per_completion', 'Yards_per_game',
    'Net_yards_per_attempt', 'Adjusted_net_yards_per_attempt'
]


QB_metrics_clean = [
    'QBR', 'Passer Rating', 'Completion Percentage', 'Yards passed in a season', 
    'TD thrown in a season','TD Percentage', 'Int thrown in a season', 'Int Percentage', 'Yards per attempt', 
    'Adjusted yards per attempt', 'Yards per completion', 'Yards per game',
    'Net yards per attempt', 'Adjusted net yards per attempt'
]

#%% 0.6 Correlation matrix

qbDataSubset = qb_data[QB_metrics]

corrMatrix = np.corrcoef(qbDataSubset, rowvar=False)

plt.figure(figsize=(10, 6))
plt.imshow(corrMatrix)
plt.colorbar()
plt.xticks(np.arange(len(QB_metrics_clean)), QB_metrics_clean, rotation=90)
plt.yticks(np.arange(len(QB_metrics_clean)), QB_metrics_clean)
plt.xlabel('QB Metric')
plt.ylabel('QB Metric')
plt.title('Correlation Matrix of QB Metrics')
plt.show()

corr_df = pd.DataFrame(corrMatrix, index=QB_metrics_clean, columns=QB_metrics_clean)

abs_corr_df = corr_df.abs()
np.fill_diagonal(abs_corr_df.values, np.nan)

pd.set_option('display.max_columns', None)

# Stack the DataFrame to get pairs and sort by correlation value
abs_corr_stack = abs_corr_df.stack().reset_index()
corr_stack = corr_df.stack().reset_index()

corr_stack.columns = ['Variable1', 'Variable2', 'Correlation']

# Remove duplicate pairs (A-B and B-A)
corr_stack['Variable1'], corr_stack['Variable2'] = np.where(
    corr_stack['Variable1'] < corr_stack['Variable2'], 
    [corr_stack['Variable1'], corr_stack['Variable2']], 
    [corr_stack['Variable2'], corr_stack['Variable1']]
)
corr_stack = corr_stack.drop_duplicates(subset=['Variable1', 'Variable2'])


abs_corr_stack.columns = ['Variable1', 'Variable2', 'Correlation']

abs_corr_stack['Variable1'], abs_corr_stack['Variable2'] = np.where(
    abs_corr_stack['Variable1'] < abs_corr_stack['Variable2'], 
    [abs_corr_stack['Variable1'], abs_corr_stack['Variable2']], 
    [abs_corr_stack['Variable2'], abs_corr_stack['Variable1']]
)
abs_corr_stack = abs_corr_stack.drop_duplicates(subset=['Variable1', 'Variable2'])

#%% 1) QB Metrics, correlation w/ winng + R^2

r_squared_values = []
corr_coefficients = []


for i, category in enumerate(QB_metrics):
    #Scatter plot
    #plt.scatter(qb_data[category], y, label='')

    # Linear regression
    x = qb_data[category]
    m, c = np.polyfit(x, y, 1)
    #plt.plot(x, m*x + c, color='red')
    
    r_squared = np.corrcoef(x, y)[0, 1]**2
    corr = np.corrcoef(x, y)[0, 1]

    r_squared_values.append(r_squared)
    corr_coefficients.append(corr)

    # Plot settings
    #plt.xlabel(QB_metrics_clean[i])
    #plt.ylabel('Win percentage')
    #plt.title(f"Win percentage vs {QB_metrics_clean[i]}")
    #plt.show()

# A plot with all of the subplots
plt.figure(figsize=(20, 8))

for i, category in enumerate(QB_metrics_clean):
    plt.subplot(2, 7, i + 1)
    plt.bar(['Corr Coeff', 'R-squared'], [corr_coefficients[i], r_squared_values[i]])
    plt.title(category)
    plt.ylim(0, 1) 
    #plt.ylim(-1, 1)
    # We can use this^ to temporarily show how Int% corr coeff is negative
    # since the rest of the numbers are positive, I chose to express it this way.
    plt.tight_layout()

plt.show()

qbStatsMatrix = np.column_stack((corr_coefficients, r_squared_values))


#%% 2) Adjusted Net Yards Per Attempt ML linear regression

x_adj_reshaped = x_Adj_n_Yds_p_att.values.reshape(-1, 1)
y_reshaped = y.values.reshape(-1, 1)

passer_x_train, passer_x_test, passer_y_train, passer_y_test = train_test_split(x_adj_reshaped, y_reshaped, test_size = 0.2, random_state = 1)

regressor = LinearRegression()
regressor.fit(passer_x_train, passer_y_train)

print("Intercept", regressor.intercept_)
print("Coefficient", regressor.coef_)

passer_y_pred = regressor.predict(passer_x_test)

# Plotting predicted vs. actual values
plt.figure(figsize=(10, 10))
plt.scatter(passer_y_test, passer_y_pred, color='gray', label='Actual vs Predicted')

line_fit = LinearRegression()
line_fit.fit(passer_y_test, passer_y_pred)

perfect_line = np.linspace(-0.05, 1.05, 100)  # Adjust range based on your data
plt.plot(perfect_line, perfect_line, color='blue', linewidth=2, label='Perfect Prediction Line (y=x)')

plt.xlabel("Actual Win %")
plt.xlim(-0.05,1.05)
plt.ylim(-0.05,1.05)
plt.ylabel("Predicted Win %")
plt.title("Actual vs Predicted Win % Based on ANY/A")
plt.legend(loc = "upper left")
plt.show()

print("Mean Absolute Error:", metrics.mean_absolute_error(passer_y_test, passer_y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(passer_y_test, passer_y_pred))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(passer_y_test, passer_y_pred)))
print("R-squared", metrics.r2_score(passer_y_test, passer_y_pred))


#%% 3) Multiple regression (all variables)

allX = qb_data[QB_metrics]
allY = qb_data['Win_Percentage']

all_X_train, all_X_test, all_y_train, all_y_test = train_test_split(allX, allY, test_size=0.2, random_state=24)

regressor = LinearRegression()
regressor.fit(all_X_train, all_y_train)

print("Intercept:", regressor.intercept_)
print("Coefficients:", regressor.coef_)

all_y_pred = regressor.predict(all_X_test)


line_fit = LinearRegression()
line_fit.fit(all_y_test.values.reshape(-1, 1), all_y_pred)

slope = line_fit.coef_[0]

# Plotting actual vs predicted values
plt.figure(figsize=(10, 10))
plt.scatter(all_y_test, all_y_pred, color='gray', label='Actual vs Predicted')

plt.plot(perfect_line, perfect_line, color='blue', linewidth=2, label='Perfect Prediction Line (y=x)')
plt.xlabel("Actual Win %")
plt.ylabel("Predicted Win %")
plt.title("Actual vs Predicted Win % Based on All QB Metrics")
plt.xlim(-0.05,1.05)
plt.ylim(-0.05,1.05)
plt.legend(loc = "upper left")
plt.show()

print("Mean Absolute Error:", metrics.mean_absolute_error(all_y_test, all_y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(all_y_test, all_y_pred))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(all_y_test, all_y_pred)))
print("R-squared", metrics.r2_score(all_y_test, all_y_pred))


# Lots of multicollinearity. Does experimenting with fewer variables yield better results?

#%% 4) Multiple regression (uncorrelated variables)

uncorrelated_QB_metrics = [

    'Cmp_Percentage', 'Yards_per_completion'
]

uncorrX = qb_data[uncorrelated_QB_metrics]

uncorr_X_train, uncorr_X_test, uncorr_y_train, uncorr_y_test = train_test_split(uncorrX, allY, test_size=0.2, random_state=8)

regressor = LinearRegression()
regressor.fit(uncorr_X_train, uncorr_y_train)

print("Intercept:", regressor.intercept_)
print("Coefficients:", regressor.coef_)

uncorr_y_pred = regressor.predict(uncorr_X_test)


line_fit = LinearRegression()
line_fit.fit(uncorr_y_test.values.reshape(-1, 1), uncorr_y_pred)

slope = line_fit.coef_[0]

# Plotting actual vs predicted values
plt.figure(figsize=(10, 10))
plt.scatter(uncorr_y_test, uncorr_y_pred, color='gray', label='Actual vs Predicted')

plt.plot(perfect_line, perfect_line, color='blue', linewidth=2, label='Perfect Prediction Line (y=x)')
plt.xlabel("Actual Win %")
plt.ylabel("Predicted Win %")
plt.title("Actual vs Predicted Win % Based on Completion percentage, Yards per completion")
plt.xlim(-0.05,1.05)
plt.ylim(-0.05,1.05)
plt.legend(loc = "upper left")
plt.show()

print("Mean Absolute Error:", metrics.mean_absolute_error(uncorr_y_test, uncorr_y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(uncorr_y_test, uncorr_y_pred))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(uncorr_y_test, uncorr_y_pred)))
print("R-squared", metrics.r2_score(uncorr_y_test,uncorr_y_pred))

#%% 5) k-means for ANY/A

scaler = StandardScaler() 
scaled_features = scaler.fit_transform(x_Adj_n_Yds_p_att.values.reshape(-1, 1)) # to save sse

error = list() # to save sse
silhouette = list() # silhouette score

#%% 5.1) Elbow method

for k in range(2,20): # silhouette requires at least 2 clusters
    kmeans = KMeans(n_clusters=k, n_init=10) # init k-means object
    kmeans.fit(scaled_features)  # run k-means!
    error.append(kmeans.inertia_) # save sum of squared error (SSE)
    
    score = metrics.silhouette_score(scaled_features, kmeans.labels_) # calc silhouette score
    silhouette.append(score) # save score

plt.figure(figsize=(12, 6))
plt.plot(range(2, len(error)+2), error, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='yellow', markersize=10)

plt.xticks(np.arange(0, 21, 1))
plt.title('K-Means: Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Squared Standard Error')
plt.show()


#%% 5.3) k-means classification graph

kmeansPasser = KMeans(
    init="random",
    n_clusters=3,
    n_init=10,
    max_iter=300,
    random_state=84
)

kmeansPasser.fit(scaled_features)
cluster_labels = kmeansPasser.labels_

# Create a new DataFrame with the cluster labels
new_df = pd.DataFrame({
    'Adjusted_net_yards_per_attempt': qb_data['Adjusted_net_yards_per_attempt'],
    'Win_Percentage': qb_data['Win_Percentage'],
    'cluster': cluster_labels
})

# Calculate the median Adjusted Net Yards Per Attempt and Win Percentage for each cluster
cluster3_df = new_df.groupby('cluster')[['Adjusted_net_yards_per_attempt', 'Win_Percentage']].median().reset_index()

cluster_colors = {0: 'red', 1: 'orange', 2: 'green'}

plt.rc('figure', figsize=(10, 6))

median_win_percentage = new_df.groupby('cluster')['Win_Percentage'].median()

for i in range(3):
    subset = new_df[new_df['cluster'] == i]
    label = f'Cluster {i + 1}'
    plt.scatter(subset['Adjusted_net_yards_per_attempt'], subset['Win_Percentage'], label=label, color=cluster_colors[i])

plt.xlabel('ANY/A')
plt.ylabel('Win Percentage')
plt.title('Win Percentage and ANY/A by Cluster')
plt.legend(loc='upper left', prop={'size': 8})
plt.show()


#%% 5.4 k-means k = 4

kmeansPasser4 = KMeans(
    init="random",
    n_clusters=4,
    n_init=10,
    max_iter=300,
    random_state=67
)

kmeansPasser4.fit(scaled_features)
cluster_labels4 = kmeansPasser4.labels_

# Create a new DataFrame with the cluster labels
new_df4 = pd.DataFrame({
    'Adjusted_net_yards_per_attempt': qb_data['Adjusted_net_yards_per_attempt'],
    'Win_Percentage': qb_data['Win_Percentage'],
    'cluster': cluster_labels4
})

# Calculate the median Adjusted Net Yards Per Attempt and Win Percentage for each cluster
cluster4_df = new_df4.groupby('cluster')[['Adjusted_net_yards_per_attempt', 'Win_Percentage']].median().reset_index()

cluster_colors4 = {0: 'red', 1: 'orange', 2: 'gold', 3: 'green'}

plt.rc('figure', figsize=(10, 6))

median_win_percentage4 = new_df4.groupby('cluster')['Win_Percentage'].median()

for i in range(4):
    subset = new_df4[new_df4['cluster'] == i]
    label = f'Cluster {i + 1}'
    plt.scatter(subset['Adjusted_net_yards_per_attempt'], subset['Win_Percentage'], label=label, color=cluster_colors4[i])

plt.xlabel('ANY/A')
plt.ylabel('Win Percentage')
plt.title('Win Percentage and ANY/A by Cluster')
plt.legend(loc='upper left', prop={'size': 8})
plt.show()

#%% 5.5 Changing cluster DFs

cluster3_df = cluster3_df.rename(columns={
    'cluster': 'Cluster',
    'Adjusted_net_yards_per_attempt': 'ANY/A',
    'Win_Percentage': 'Win Percentage'
})
cluster3_df.iloc[0, 0] = 1
cluster3_df.iloc[1, 0] = 2
cluster3_df.iloc[2, 0] = 3


cluster4_df = cluster4_df.rename(columns={
    'cluster': 'Cluster',
    'Adjusted_net_yards_per_attempt': 'ANY/A',
    'Win_Percentage': 'Win Percentage'
})

cluster4_df.iloc[0, 0] = 1
cluster4_df.iloc[1, 0] = 2
cluster4_df.iloc[2, 0] = 3
cluster4_df.iloc[3, 0] = 4


#%% 6) KNN

# Revisualizing our data with the three most conventionally accepted tiers of QBs:
# Starters (no honors), Pro Bowl (All Stars), All-Pro ("Best" QB in that season)
# Pro Bowl and All-Pro have different selection criteria:
# Pro Bowl by the coaches,  players, and fans vote. Each group's ballots count for ⅓, or 33.3%.
# All-Pros are selected by members of the media.

custom_rank_names = {0: 'No honors', 1: 'Pro Bowl', 2: 'All-Pro'}

kNNxPasser = x_Adj_n_Yds_p_att.values.reshape(-1, 1)
kNNyPasser = y.values

ranking = qb_data['Rank']

label_encoder = LabelEncoder()
ranking_encoded = label_encoder.fit_transform(ranking)

knn_x_train, knn_x_test, knn_y_train, knn_y_test = train_test_split(kNNxPasser, ranking_encoded, test_size=0.2, random_state=24)

knnPasser = KNeighborsClassifier(n_neighbors=7)
knnPasser.fit(knn_x_train, knn_y_train)

knn_y_pred = knnPasser.predict(knn_x_test)

cmap = plt.get_cmap('copper')
colors = [cmap(i) for i in np.linspace(0, 1, len(label_encoder.classes_))]

for i, rank in enumerate(label_encoder.classes_):
    idx = ranking_encoded == i
    plt.scatter(kNNxPasser[idx], kNNyPasser[idx], color=colors[i], label=custom_rank_names[i])

plt.xlabel('ANY/A')
plt.ylabel('Win percentage')
plt.title('Win Percentage and ANY/A by Cluster')
plt.legend(title='Rank')
plt.show()


medianValues = qb_data.groupby("Rank")[["Rank", "Adjusted_net_yards_per_attempt", "Win_Percentage"]].median()
medianValues.at[0, 'Rank'] = "No honors"
medianValues.at[1, 'Rank'] = "Pro Bowl"
medianValues.at[2, 'Rank'] = "All-Pro"

medianValues = medianValues.rename(columns={
    'Adjusted_net_yards_per_attempt': 'ANY/A',
    'Win_Percentage': 'Win Percentage'
})

#%% Counting ranks in each cluster

new_df4 = pd.DataFrame({
    'Adjusted_net_yards_per_attempt': qb_data['Adjusted_net_yards_per_attempt'],
    'Win_Percentage': qb_data['Win_Percentage'],
    'Rank': qb_data['Rank'],
    'cluster': cluster_labels4
})

# Store the counts of each rank in each cluster
rank_clusters = pd.DataFrame(columns=['No honors', 'Pro Bowl', 'All-Pro'], index=range(4)).fillna(0).astype(int)

for cluster in range(4):
    cluster_data = new_df4[new_df4['cluster'] == cluster]
    rank_counts = cluster_data['Rank'].value_counts().sort_index()
    for rank in rank_counts.index:
        if rank == 0:
            rank_clusters.at[cluster, 'No honors'] = rank_counts[rank]
        elif rank == 1:
            rank_clusters.at[cluster, 'Pro Bowl'] = rank_counts[rank]
        elif rank == 2:
            rank_clusters.at[cluster, 'All-Pro'] = rank_counts[rank]

rank_columns = ['K-means cluster'] + [col for col in rank_clusters.columns if col != 'K-means cluster']
rank_clusters = rank_clusters.reindex(columns=rank_columns)

rank_clusters.at[0, 'K-means cluster'] = "Terrible"
rank_clusters.at[1, 'K-means cluster'] = "Bad"
rank_clusters.at[2, 'K-means cluster'] = "Good"
rank_clusters.at[3, 'K-means cluster'] = "Great"


print(rank_clusters)

#%%

# List of k values to try
k_values = [3, 5, 7, 9, 11, 13, 15]
accuracies = []

for k in k_values:
    knnPasser = KNeighborsClassifier(n_neighbors=k)
    knnPasser.fit(knn_x_train, knn_y_train)
    knn_y_pred = knnPasser.predict(knn_x_test)
    accuracy = (knn_y_pred == knn_y_test).sum() / len(knn_y_test)
    accuracies.append(accuracy)

# Plot the accuracies
plt.figure(figsize=(10, 6))
plt.bar(k_values, accuracies, color='skyblue')
plt.xlabel('k Value')
plt.ylabel('Accuracy')
plt.title('Accuracy for Different k Values in k-NN')
plt.xticks(k_values)
plt.show()


