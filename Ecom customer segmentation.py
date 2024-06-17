
#Import the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


# These lines import various libraries:
# 
# pandas: for data manipulation and analysis.
# 
# numpy: for numerical operations.
# 
# seaborn and matplotlib: for data visualization.
# 
# sklearn: for machine learning tasks like scaling and clustering.
# 
# yellowbrick: for visualizing the elbow method to determine the optimal number of clusters.



# Mount Google Drive to access the file
from google.colab import drive
drive.mount('/content/drive')


# Define file path and load the data
file_path = '/content/drive/My Drive/Colab Notebooks/ecom customer_data.xlsx'
data = pd.read_excel(file_path)


data=pd.read_excel("/content/drive/My Drive/Colab Notebooks/ecom customer_data.xlsx")
data


# These lines mount Google Drive to access files stored there and load an Excel file into a pandas DataFrame. The head() method displays the first few rows of the dataset.


data.head()


# The data.head() method displays the first few rows of the dataset.


df=data.copy()
df.info()


# gives an overview of the dataset, including the number of non-null entries in each column, which helps identify missing values and understand data types.

# Create a copy of the dataset.


df.describe()


# Provide statistical summaries of numerical columns.

# **Data Cleaning**


#Check the duplicates
df[df.duplicated()]


# check if there is duplicate value in the dataset

df.isna().sum()


# Check for missing values


df['Gender']=df['Gender'].fillna(df['Gender'].mode()[0])


# Fill missing values in the 'Gender' column with the most frequent value (mode).


df.isna().sum().sum()


# Check again to see if there is missing value

# **Data Visualization**




df.Gender.value_counts()


# Shows the distribution of genders, which is important for understanding the demographic makeup of the customers.



sns.countplot(data=df,x='Gender')
plt.show()


# Plot of the genders




plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(data=df,x='Orders')


# We visualize the "orders" column




#Order count by each number
plt.subplot(1,2,2)
sns.countplot(data=df,x='Orders',hue='Gender')
plt.suptitle("Overall Orders VS Gender wise Orders")
plt.show()


# Visualizing orders gender-wise




#Orders and searches of each brands
cols=list(df.columns[2:])
def dist_list(lst):
  plt.figure(figsize=(30,30))
  for i, col in enumerate(lst,1):
    plt.subplot(6,6,i)
    sns.boxplot(data=df,x=df[col])
dist_list(cols)


# Visualizing each feature with a box plot helps identify the distribution, median, quartiles, and potential outliers. This is useful for understanding the spread and range of each feature.




#Heatmap
plt.figure(figsize=(20,15))
sns.heatmap(df.iloc[:,3:].corr())
plt.show()


# A correlation heatmap shows how features are related to each other. High correlation between features might indicate redundancy, which can affect clustering.




df.iloc[:2,:].hist(figsize=(40,30))
plt.show()


# Displaying histograms for the first few rows helps understand the distribution of values in these rows, which might highlight any peculiarities or anomalies.

# **Feature Engineering**




new_df=df.copy()
new_df['Total Search']=new_df.iloc[:,3:].sum(axis=1)


# Create a new DataFrame new_df and add a 'Total Search' column summing up search counts below.




new_df.sort_values('Total Search', ascending=False)


#  Sorts the DataFrame new_df based on the values in the 'Total Search' column in descending order, meaning it arranges rows from the highest to the lowest values of 'Total Search'.




plt.figure(figsize=(13,8))
plt_data=new_df.sort_values('Total Search',ascending=False)[['Cust_ID','Gender','Total Search']].head(10)
sns.barplot(data=plt_data,
            x='Cust_ID',
            y='Total Search',
            hue='Gender',
            order=plt_data.sort_values('Total Search',ascending=False).Cust_ID)
plt.title("Top 10 Cust_ID based on Total Searches")
plt.show()


# Create a bar plot of the top 10 customers based on total searches, colored by gender.

# **Scaling**




x=df.iloc[:,2:].values
x


# Extract features from the dataset (excluding the first two columns).




scale=MinMaxScaler()
features=scale.fit_transform(x)
features


# Scale these features to a range between 0 and 1 using MinMaxScaler

# **Clustering with k-means by finding optimal number of clusters**

# Elbow method to get the optimal K value




inertia=[]
for i in range(1,16):
  k_means=KMeans(n_clusters=i)
  k_means=k_means.fit(features)
  inertia.append(k_means.inertia_)


# Purpose of this code is to evaluate the appropriate number of clusters (n_clusters) for K-means clustering by comparing how inertia changes with different cluster numbers.
# Inertia is typically used as a criterion to select the optimal number of clusters. A lower inertia suggests better clustering (more compact clusters).
# By plotting the values in the inertia list against the number of clusters (range(1, 16)), you can visually identify the "elbow" point where the inertia starts to decrease more slowly, indicating the optimal number of clusters.




#Elbow graph
plt.figure(figsize=(20,7))
plt.subplot(1,2,1)
plt.plot(range(1,16),inertia, 'bo-')
plt.xlabel('No of clusters'),plt.ylabel('Inertia')


# Use the elbow method to find the optimal number of clusters.
# The elbow method involves plotting the inertia (sum of squared distances to cluster centers) for different numbers of clusters.




#Kelbow visualizer
plt.subplot(1,2,2)
kmeans=KMeans()
visualize=KElbowVisualizer(kmeans,k=(1,16))
visualize.fit(features)
plt.suptitle("Elbow Graph and Elbow Visualizer")
visualize.poof()
plt.show()


# Yellowbrick's KElbowVisualizer automates the elbow method visualization, making it easier to identify the optimal number of clusters.

# **Silhouette Score for each k value**




silhouette_avg=[]
for i in range(2,16):
  #initialize kmeans cluster
  kmeans=KMeans(n_clusters=i)
  cluster_labels=kmeans.fit_predict(features)
  #Silhouette score
  silhouette_avg.append(silhouette_score(features,cluster_labels))


# This score measures how similar an object is to its own cluster compared to other clusters. A higher silhouette score indicates better-defined clusters.Calculate the silhouette score for different numbers of clusters.




plt.figure(figsize=(10,7))
plt.plot(range(2,16),silhouette_avg, 'bX-')
plt.xlabel('Values of K')
plt.ylabel('Silhouette score')
plt.title('Silhouette analysis for optimal K')
plt.show()


# Plot the silhouette scores to help determine the optimal number of clusters.

# **Kmeans Model Here we will take K value as 3 as per Elbow Method**

# Building the Final KMeans Model




model=KMeans(n_clusters=3)
model=model.fit(features)

y_km=model.predict(features)
centers=model.cluster_centers_

df['Cluster']=pd.DataFrame(y_km)
df.to_csv("Cluster_data", index=False)

df["Cluster"].value_counts()



# Initialize the KMeans model with a specified number of clusters (n_clusters).
# Fit the model to the data (features).
# 
# Predict cluster labels for each data point.
# 
# Analyze and manipulate the clustered data, such as adding cluster labels to the original dataset (df), saving the results to a file, and exploring cluster distributions (cluster_counts).

# **Analyzing and Visualizing Clusters**




sns.countplot(data=df,x='Cluster')
plt.show()


# Visualizing the number of customers in each cluster helps understand the distribution and size of each segment.




c_df=pd.read_csv('Cluster_data')


# Re-loading the cluster data




c_df.head()


# This helps us get the first few data of the cluster data




c_df['Total Search']=c_df.iloc[:,3:38].sum(axis=1)


# Calculating the 'Total Search' feature for further analysis of each cluster.

# 
# 
# ```
# # This is formatted as code
# ```
# 
# Analyze the cluster 0




cl_0=c_df.groupby(['Cluster','Gender'], as_index=False).sum().query('Cluster==0')


# Load the clustered data.
# Analyze Cluster 0, showing customer count and total searches by gender.




cl_0





plt.figure(figsize=(15,6))

plt.subplot(1,2,1)

sns.countplot(data=c_df.query('Cluster==0'),x='Gender')

plt.title('Customers count')



plt.subplot(1,2,2)

sns.barplot(data=cl_0,x='Gender',y='Total Search')

plt.title('Total Searches by Gender')

plt.suptitle('No. of customer and their total searches in "Cluster 0"')

plt.show()


# Analyze the cluster 1




cl_1=c_df.groupby(['Cluster','Gender'], as_index=False).sum().query('Cluster==1')


# Load the clustered data.
# Analyze Cluster 1, showing customer count and total searches by gender.




cl_1





plt.figure(figsize=(15,6))

plt.subplot(1,2,1)

sns.countplot(data=c_df.query('Cluster==1'),x='Gender')

plt.title('Customers count')



plt.subplot(1,2,2)

sns.barplot(data=cl_1,x='Gender',y='Total Search')

plt.title('Total Searches by Gender')

plt.suptitle('No. of customer and their total searches in "Cluster 1"')

plt.show()


# Analyse Cluster 2




cl_2=c_df.groupby(['Cluster','Gender'], as_index=False).sum().query('Cluster==2')


# Load the clustered data.
# Analyze Cluster 2, showing customer count and total searches by gender.




cl_2





plt.figure(figsize=(15,6))

plt.subplot(1,2,1)

sns.countplot(data=c_df.query('Cluster==2'),x='Gender')

plt.title('Customers count')



plt.subplot(1,2,2)

sns.barplot(data=cl_1,x='Gender',y='Total Search')

plt.title('Total Searches by Gender')

plt.suptitle('No. of customer and their total searches in "Cluster 2"')

plt.show()


# Overall Analysis




final_df=c_df.groupby(['Cluster'], as_index=False).sum()
final_df





plt.figure(figsize=(15,6))

sns.countplot(data=c_df,x='Cluster',hue='Gender')

plt.title('Total Customers on each Cluster')

plt.show()


# Visualize the total customers ON EACH CLUSTER




plt.figure(figsize=(15,6))

plt.subplot(1,2,1)

sns.barplot(data=final_df,x='Cluster',y='Total Search')

plt.title('Total Searches by each Group')

plt.subplot(1,2,2)

sns.barplot(data=final_df,x='Cluster',y='Orders')

plt.title('Past Orders by each group')

plt.suptitle('No of times customer searched the products and their past orders')

plt.show()



# Visualises total searches by group and past orders by each group, helping to draw conclusions about each customer segment's behavior and preferences.
