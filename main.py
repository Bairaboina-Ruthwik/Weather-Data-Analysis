import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("weather_classification_data.csv")

print(df.head())
print(df.info())
print(df.describe())

print(df.isnull().sum())

df = df.dropna()

plt.figure()
sns.histplot(data =df,x="Temperature" ,kde = True,hue= "Weather Type",palette="dark")
plt.title("Temperature Distribution")
plt.savefig("outputs/temperature_distribution.png")
plt.show()


plt.figure()
sns.histplot(data =df,x="Humidity" ,kde = True,hue= "Weather Type",palette="Set1")
plt.title("Humidity Distribution")
plt.savefig("outputs/humidity_distribution.png")
plt.show()


plt.figure()
sns.scatterplot(x="Temperature",y="Humidity",hue= "Weather Type",data=df)
plt.title("Temperature vs Humidity")
plt.legend(title="Weather Type")
plt.savefig("outputs/temp_vs_humidity.png")
plt.show()

plt.figure()
sns.scatterplot(x="Temperature",y="Wind Speed",hue="Weather Type",data=df)
plt.title("Temperature vs Wind Speed")
plt.legend(title="Weather Type")
plt.savefig("outputs/temp_vs_windspeed.png")
plt.show()

plt.figure()
sns.regplot(x="Temperature", y="Humidity",color="purple",data=df,)
plt.title("Temperature vs Humidity (Trend)")
plt.savefig("outputs/regression_temp_humidity.png")
plt.show()

plt.figure()
sns.violinplot(x="Weather Type",y="Temperature",hue="Weather Type",data=df,palette="Set2")
plt.xticks(rotation=45)
plt.title("Temperature Distribution by Weather Type")
plt.savefig("outputs/violin_temp_weather.png")
plt.show()

plt.figure()
sns.countplot(x="Cloud Cover", hue="Weather Type", data=df, palette="Set2")
plt.title("Cloud Cover vs Weather Type")
plt.legend(title="Weather Type")
plt.savefig("outputs/cloud_cover_vs_weather.png")
plt.show()

plt.figure()
sns.countplot(x="Season", hue="Weather Type", data=df, palette="Set2")
plt.title("Season vs Weather Type")
plt.legend(title="Weather Type")
plt.savefig("outputs/season_vs_weather.png")
plt.show()

plt.figure()
sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.title("Correlation Heatmap")
plt.savefig("outputs/correlation_heatmap.png")
plt.show()

plt.figure()
sns.boxplot(x="Weather Type", y="Precipitation (%)",hue ="Weather Type", data=df, palette="Set2")
plt.title("Precipitation vs Weather Type")
plt.savefig("outputs/precipitation_boxplot.png")
plt.show()

weather_counts = df["Weather Type"].value_counts()

plt.figure()
sns.barplot(x="Weather Type",y="Temperature",hue="Weather Type",data=df,palette="Set2")
plt.title("Average Temperature by Weather Type")
plt.savefig("outputs/avg_temp_barplot.png")
plt.show()
