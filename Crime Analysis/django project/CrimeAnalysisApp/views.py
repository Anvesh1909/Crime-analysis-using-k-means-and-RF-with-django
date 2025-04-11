from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
import matplotlib
matplotlib.use('Agg', force=True)  # Force the Agg backend
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # Ensure we're using Agg backend
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
import traceback

global dataset, kmeans_cluster, theft_cls, rape_cls, murder_cls

sc = MinMaxScaler(feature_range = (0, 1))
le1 = LabelEncoder()
le2 = LabelEncoder()

le3 = LabelEncoder()
le4 = LabelEncoder()

global mse, rmse

def calculateError(alg, X_test, y_test):
    predict = alg.predict(X_test)
    #predict = predict.reshape(predict.shape[0],1)
    #predict = sc.inverse_transform(predict)
    predict = predict.ravel()
    #labels = sc.inverse_transform(y_test)
    labels = y_test.ravel()
    mse_error = mean_squared_error(labels,predict)
    rmse_error = sqrt(mse_error)
    mse.append(mse_error/1000)
    rmse.append(rmse_error)

def UploadDatasetAction(request):
    if request.method == 'POST':
        global dataset, kmeans_cluster, theft_cls, rape_cls, murder_cls, mse, rmse
        mse = []
        rmse = []
        myfile = request.FILES['t1']
        dataset = pd.read_csv("Dataset/cleaned_crime_data.csv", usecols=['States/UTs','District', 'Murder', 'RAPE', 'THEFT', 'DOWRY_DEATHS', 'Year'])
        dataset.fillna(0, inplace = True)
        cols = ['States/UTs', 'District']
        dataset[cols[0]] = pd.Series(le1.fit_transform(dataset[cols[0]].astype(str)))
        dataset[cols[1]] = pd.Series(le2.fit_transform(dataset[cols[1]].astype(str)))
        X = dataset.values
        X = sc.fit_transform(X)
        kmeans_cluster = KMeans(n_clusters=2, n_init=1200)
        kmeans_cluster.fit(X)

        dataset = pd.read_csv("Dataset/cleaned_crime_data.csv", usecols=['States/UTs','District', 'Year', 'THEFT', 'Murder', 'RAPE'])
        dataset.fillna(0, inplace = True)
        print(dataset)
        cols = ['States/UTs', 'District']
        dataset[cols[0]] = pd.Series(le3.fit_transform(dataset[cols[0]].astype(str)))
        dataset[cols[1]] = pd.Series(le4.fit_transform(dataset[cols[1]].astype(str)))
        theft_Y = dataset['THEFT'].values
        murder_Y = dataset['Murder'].values
        rape_Y = dataset['RAPE'].values
        dataset.drop(['THEFT'], axis = 1,inplace=True)
        dataset.drop(['Murder'], axis = 1,inplace=True)
        dataset.drop(['RAPE'], axis = 1,inplace=True)
        X = dataset.values

        X_train1, X_test1, y_train1, y_test1 = train_test_split(X, theft_Y, test_size = 0.2)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X, rape_Y, test_size = 0.2)
        X_train3, X_test3, y_train3, y_test3 = train_test_split(X, murder_Y, test_size = 0.2)

        theft_cls = RandomForestRegressor()
        theft_cls.fit(X, theft_Y)
        calculateError(theft_cls, X_test1, y_test1)

        rape_cls = RandomForestRegressor()
        rape_cls.fit(X, rape_Y)
        calculateError(rape_cls, X_test2, y_test2)

        murder_cls = RandomForestRegressor()
        murder_cls.fit(X, murder_Y)
        calculateError(murder_cls, X_test3, y_test3)
        
        dataset = pd.read_csv("Dataset/cleaned_crime_data.csv")
        dataset.fillna(0, inplace = True)
        columns = list(dataset.columns)
        strdata = '<table border=1 align=center width=100%><tr><th><font size="" color="black">'+columns[0]+'</th>'
        for i in range(1,len(columns)):
            strdata+='<th><font size="" color="black">'+columns[i]+'</th>'
        strdata += "</tr>"
        dataset = dataset.values
        for i in range(len(dataset)):
            strdata += "<tr>"
            for j in range(len(dataset[i])):
                strdata+='<td><font size="" color="black">'+str(dataset[i,j])+'</td>'
            strdata += "</tr>"
        context= {'data':strdata}            
        return render(request, 'ViewDataset.html', context)                    
    
def AdminLogin(request):
    if request.method == 'POST':
        user = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        if user == 'admin' and password == 'admin':
            context= {'data':user}            
            return render(request, 'AdminScreen.html', context)
        else:
            context= {'data':"Invalid login details"}            
            return render(request, 'Admin.html', context)            
        
def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Admin(request):
    if request.method == 'GET':
       return render(request, 'Admin.html', {})

def UploadDataset(request):
    if request.method == 'GET':
       return render(request, 'UploadDataset.html', {})

def MSEGraph(request):
    if request.method == 'GET':
        # Initialize rmse and mse if they don't exist
        global rmse, mse
        if 'rmse' not in globals():
            rmse = [0, 0, 0]  # Initialize with default values
        if 'mse' not in globals():
            mse = [0, 0, 0]   # Initialize with default values

        strdata = '<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">RMSE</th>'
        strdata+='<th><font size="" color="black">MSE</th></tr>'
        strdata+='<tr><td><font size="" color="black">Random Forest Theft Prediction</td><td><font size="" color="black">'+str(rmse[0])+'</td>'
        strdata+='<td><font size="" color="black">'+str(mse[0])+'</td>'
        strdata+='<tr><td><font size="" color="black">Random Forest Rape Prediction</td><td><font size="" color="black">'+str(rmse[1])+'</td>'
        strdata+='<td><font size="" color="black">'+str(mse[1])+'</td>'
        strdata+='<tr><td><font size="" color="black">Random Forest Murder Prediction</td><td><font size="" color="black">'+str(rmse[2])+'</td>'
        strdata+='<td><font size="" color="black">'+str(mse[2])+'</td></tr></table>'

        df = pd.DataFrame([['Random Forest Theft Prediction','RMSE',rmse[0]],['Random Forest Theft Prediction','MSE',mse[0]],
                           ['Random Forest Rape Prediction','RMSE',rmse[1]],['Random Forest Rape Prediction','MSE',mse[1]],
                           ['Random Forest Murder Prediction','RMSE',rmse[2]],['Random Forest Murder Prediction','MSE',mse[2]],
                          ],columns=['Parameters','Algorithms','Value'])
        
        # Create pivot table using pivot_table instead of pivot
        pivot_df = df.pivot_table(index='Parameters', columns='Algorithms', values='Value', aggfunc='first')
        
        # Create the plot and save it to a file
        plt.figure(figsize=(12, 8))
        ax = pivot_df.plot(kind='bar', rot=45)
        plt.title('Model Performance Comparison', pad=20)
        plt.ylabel('Error Value')
        plt.xlabel('Prediction Model')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add value labels on top of each bar
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', 
                       xytext=(0, 10), 
                       textcoords='offset points')
        
        # Save the plot to a file
        plot_path = os.path.join('static', 'analysis', 'mse_graph.png')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Add the image to the response
        strdata += '<tr><td colspan="3" class="text-center"><img src="/static/analysis/mse_graph.png" class="img-fluid" style="max-width:100%;"/></td></tr>'
        
        context= {'data':strdata}
        return render(request, 'Comparison.html', context)
                                                        
                                                                                                                               

def ClusterPrediction(request):
    if request.method == 'GET':
        dataset = pd.read_csv("Dataset/cleaned_crime_data.csv", usecols=['States/UTs','District', 'Year'])
        dataset.fillna(0, inplace = True)
        
        # Get unique states
        states = np.unique(dataset['States/UTs'].values)
        states_html = '<option value="">Select State</option>'
        for state in states:
            states_html += f'<option value="{state}">{state}</option>'
        
        # Get unique districts
        districts = np.unique(dataset['District'].values)
        districts_html = '<option value="">Select District</option>'
        for district in districts:
            districts_html += f'<option value="{district}">{district}</option>'
        
        # Get unique years
        years = np.unique(dataset['Year'].values)
        years_html = '<option value="">Select Year</option>'
        for year in years:
            years_html += f'<option value="{year}">{year}</option>'
        
        context = {
            'states': states_html,
            'districts': districts_html,
            'years': years_html
        }
        return render(request, 'ClusterPrediction.html', context)

def ClusterPredictionAction(request):
    if request.method == 'POST':
        # Get form data with proper validation
        state = request.POST.get('t1', '')
        district = request.POST.get('t2', '')
        year = request.POST.get('t3', '')
        murder = request.POST.get('t4', '0')
        rape = request.POST.get('t5', '0')
        theft = request.POST.get('t6', '0')
        dowry = request.POST.get('t7', '0')
        
        # Validate required fields
        if not state or not district or not year:
            context = {'data': "Please fill in all required fields (State, District, and Year)"}
            return render(request, 'index.html', context)
            
        try:
            # Convert numeric fields to integers
            murder = int(murder) if murder else 0
            rape = int(rape) if rape else 0
            theft = int(theft) if theft else 0
            dowry = int(dowry) if dowry else 0
            year = int(year)
            
            # Load dataset
            dataset = pd.read_csv("Dataset/cleaned_crime_data.csv")
            dataset.fillna(0, inplace = True)
            
            # Calculate averages for comparison
            avg_murder = dataset['Murder'].mean()
            avg_rape = dataset['RAPE'].mean()
            avg_theft = dataset['THEFT'].mean()
            avg_dowry = dataset['DOWRY_DEATHS'].mean()
            
            # Fit LabelEncoders with all possible values
            le1.fit(dataset['States/UTs'].astype(str))
            le2.fit(dataset['District'].astype(str))
            
            # Handle unseen labels by adding them to the encoders
            if state not in le1.classes_:
                le1.classes_ = np.append(le1.classes_, state)
            if district not in le2.classes_:
                le2.classes_ = np.append(le2.classes_, district)
            
            # Prepare data for clustering
            cols = ['States/UTs', 'District']
            dataset[cols[0]] = pd.Series(le1.transform(dataset[cols[0]].astype(str)))
            dataset[cols[1]] = pd.Series(le2.transform(dataset[cols[1]].astype(str)))
            
            X = dataset[['States/UTs', 'District', 'Murder', 'RAPE', 'THEFT', 'DOWRY_DEATHS', 'Year']].values
            sc.fit(X)
            
            # Train KMeans
            kmeans_cluster = KMeans(n_clusters=2, n_init=1200)
            kmeans_cluster.fit(X)
            
            # Prepare prediction data
            temp = []
            temp.append([state, district, murder, rape, theft, dowry, year])
            test = pd.DataFrame(temp, columns=['States/UTs','District', 'Murder', 'RAPE', 'THEFT', 'DOWRY_DEATHS', 'Year'])
            test.fillna(0, inplace = True)
            
            # Transform the input data
            test['States/UTs'] = pd.Series(le1.transform(test['States/UTs'].astype(str)))
            test['District'] = pd.Series(le2.transform(test['District'].astype(str)))
            test = test.values
            test = sc.transform(test)
            
            predict = kmeans_cluster.predict(test)
            is_high_crime = predict[0] == 1
            
            # Prepare detailed analysis
            crime_details = [
                {
                    'type': 'Murder',
                    'current': murder,
                    'average': round(avg_murder, 2),
                    'is_above_average': murder > avg_murder,
                    'status': 'Above Average' if murder > avg_murder else 'Below Average'
                },
                {
                    'type': 'Rape',
                    'current': rape,
                    'average': round(avg_rape, 2),
                    'is_above_average': rape > avg_rape,
                    'status': 'Above Average' if rape > avg_rape else 'Below Average'
                },
                {
                    'type': 'Theft',
                    'current': theft,
                    'average': round(avg_theft, 2),
                    'is_above_average': theft > avg_theft,
                    'status': 'Above Average' if theft > avg_theft else 'Below Average'
                },
                {
                    'type': 'Dowry Deaths',
                    'current': dowry,
                    'average': round(avg_dowry, 2),
                    'is_above_average': dowry > avg_dowry,
                    'status': 'Above Average' if dowry > avg_dowry else 'Below Average'
                }
            ]
            
            # Generate recommendations based on analysis
            recommendations = []
            if is_high_crime:
                recommendations.append("This area is classified as a high crime rate zone.")
                recommendations.append("Consider increasing police presence and surveillance.")
                recommendations.append("Implement community awareness programs.")
            else:
                recommendations.append("This area is classified as a low crime rate zone.")
                recommendations.append("Maintain current security measures.")
                recommendations.append("Continue community engagement programs.")
            
            # Add specific recommendations based on crime types
            for crime in crime_details:
                if crime['is_above_average']:
                    recommendations.append(f"Focus on reducing {crime['type'].lower()} cases through targeted interventions.")
            
            prediction_result = {
                'is_high_crime': is_high_crime,
                'message': f"{district} is classified as a {'High' if is_high_crime else 'Low'} Crime Rate Area",
                'crime_details': crime_details,
                'recommendations': recommendations
            }
            
            context = {'prediction_result': prediction_result}
            return render(request, 'ClusterPrediction.html', context)
            
        except Exception as e:
            context = {'data': f"An error occurred: {str(e)}"}
            return render(request, 'index.html', context)

def FuturePrediction(request):
    if request.method == 'GET':
        dataset = pd.read_csv("Dataset/cleaned_crime_data.csv", usecols=['States/UTs','District', 'Year'])
        dataset.fillna(0, inplace = True)
        
        # Get unique states
        states = np.unique(dataset['States/UTs'].values)
        states_html = '<option value="">Select State</option>'
        for state in states:
            states_html += f'<option value="{state}">{state}</option>'
        
        # Get unique districts
        districts = np.unique(dataset['District'].values)
        districts_html = '<option value="">Select District</option>'
        for district in districts:
            districts_html += f'<option value="{district}">{district}</option>'
        
        # Get the latest year from dataset
        latest_year = int(dataset['Year'].max())
        
        # Create future years options (next 5 years)
        years_html = '<option value="">Select Year</option>'
        for year in range(latest_year + 1, latest_year + 6):
            years_html += f'<option value="{year}">{year}</option>'
        
        context = {
            'states': states_html,
            'districts': districts_html,
            'years': years_html
        }
        return render(request, 'FuturePrediction.html', context)

def FuturePredictionAction(request):
    if request.method == 'POST':
        state = request.POST.get('t1', '')
        district = request.POST.get('t2', '')
        year = request.POST.get('t3', '')
        classify_type = request.POST.get('t4', '')
        
        if not all([state, district, year, classify_type]):
            context = {'data': "Please fill in all required fields"}
            return render(request, 'index.html', context)
            
        try:
            # Load dataset
            dataset = pd.read_csv("Dataset/cleaned_crime_data.csv")
            dataset.fillna(0, inplace = True)
            
            # Get historical data for the selected district
            district_data = dataset[
                (dataset['States/UTs'] == state) & 
                (dataset['District'] == district)
            ].sort_values('Year')
            
            # Fit LabelEncoders with all possible values
            le1.fit(dataset['States/UTs'].astype(str))
            le2.fit(dataset['District'].astype(str))
            
            # Handle unseen labels by adding them to the encoders
            if state not in le1.classes_:
                le1.classes_ = np.append(le1.classes_, state)
            if district not in le2.classes_:
                le2.classes_ = np.append(le2.classes_, district)
            
            # Prepare data for prediction
            cols = ['States/UTs', 'District']
            dataset[cols[0]] = pd.Series(le1.transform(dataset[cols[0]].astype(str)))
            dataset[cols[1]] = pd.Series(le2.transform(dataset[cols[1]].astype(str)))
            
            # Prepare features and target variables
            X = dataset[['States/UTs', 'District', 'Year']].values
            sc.fit(X)
            
            # Train Random Forest models for each crime type
            theft_Y = dataset['THEFT'].values
            murder_Y = dataset['Murder'].values
            rape_Y = dataset['RAPE'].values
            
            # Train models
            theft_cls = RandomForestRegressor()
            theft_cls.fit(X, theft_Y)
            
            murder_cls = RandomForestRegressor()
            murder_cls.fit(X, murder_Y)
            
            rape_cls = RandomForestRegressor()
            rape_cls.fit(X, rape_Y)
            
            # Select appropriate model based on crime type
            if classify_type == "Theft":
                model = theft_cls
                historical_values = district_data['THEFT'].values
            elif classify_type == "Murder":
                model = murder_cls
                historical_values = district_data['Murder'].values
            else:  # Rape
                model = rape_cls
                historical_values = district_data['RAPE'].values
            
            # Prepare prediction data
            temp = []
            temp.append([state, district, int(year)])
            test = pd.DataFrame(temp, columns=['States/UTs','District', 'Year'])
            test.fillna(0, inplace = True)
            
            # Transform the input data
            test['States/UTs'] = pd.Series(le1.transform(test['States/UTs'].astype(str)))
            test['District'] = pd.Series(le2.transform(test['District'].astype(str)))
            test = test.values
            
            # Make prediction
            predicted_value = int(model.predict(test)[0])
            
            # Calculate confidence level based on historical data variance
            confidence_level = min(95, max(60, 100 - (np.std(historical_values) / np.mean(historical_values) * 100)))
            uncertainty_level = 100 - confidence_level
            
            # Determine trend direction
            if len(historical_values) >= 2:
                trend = np.polyfit(range(len(historical_values)), historical_values, 1)[0]
                trend_direction = "Increasing" if trend > 0 else "Decreasing"
            else:
                trend_direction = "Stable"
            
            # Generate prediction line for visualization
            historical_years = district_data['Year'].values.tolist()
            historical_years.append(int(year))
            prediction_line = historical_values.tolist()
            prediction_line.append(predicted_value)
            
            # Prepare factors affecting prediction
            factors = [
                {
                    'name': 'Historical Trend',
                    'description': f'The {trend_direction.lower()} trend in {classify_type.lower()} cases',
                    'impact': 'positive' if trend_direction == "Decreasing" else 'negative'
                },
                {
                    'name': 'Data Consistency',
                    'description': 'How consistent the historical data is',
                    'impact': 'positive' if confidence_level > 80 else 'negative'
                },
                {
                    'name': 'Time Gap',
                    'description': f'Time difference from last recorded data',
                    'impact': 'negative' if int(year) - max(historical_years) > 2 else 'positive'
                }
            ]
            
            # Generate recommendations
            recommendations = []
            if predicted_value > np.mean(historical_values):
                recommendations.append(f"Expected increase in {classify_type.lower()} cases. Consider preventive measures.")
            else:
                recommendations.append(f"Expected decrease in {classify_type.lower()} cases. Maintain current measures.")
            
            if confidence_level < 80:
                recommendations.append("Low confidence in prediction. Consider gathering more recent data.")
            
            if trend_direction == "Increasing":
                recommendations.append("Implement additional security measures to counter the rising trend.")
            
            prediction_details = {
                'predicted_value': predicted_value,
                'prediction_interpretation': f"Expected {predicted_value} {classify_type.lower()} cases in {year}",
                'confidence_level': round(confidence_level, 2),
                'confidence_interpretation': f"{round(confidence_level, 2)}% confidence in this prediction",
                'trend_direction': trend_direction,
                'trend_interpretation': f"The trend is {trend_direction.lower()} based on historical data",
                'historical_years': historical_years,
                'historical_values': historical_values.tolist(),
                'prediction_line': prediction_line,
                'uncertainty_level': round(uncertainty_level, 2),
                'factors': factors,
                'recommendations': recommendations
            }
            
            context = {
                'prediction': f"Future Predicted {classify_type}s = {predicted_value}",
                'prediction_details': prediction_details
            }
            return render(request, 'FuturePrediction.html', context)
            
        except Exception as e:
            context = {'data': f"An error occurred: {str(e)}"}
            return render(request, 'index.html', context)

def Analysis(request):
    if request.method == 'GET':
       return render(request, 'Analysis.html', {})

def AnalysisAction(request):
    if request.method == 'POST':
        classify_type = request.POST.get('t1', False)
        strdata = '''
        <div class="container-fluid p-0">
            <div class="row m-0">
                <div class="col-12 p-4">
        '''
        
        try:
            # Load the dataset
            dataset = pd.read_csv("Dataset/cleaned_crime_data.csv")
            
            # Create directory if it doesn't exist
            os.makedirs('static/analysis', exist_ok=True)
            
            if classify_type == "Theft":
                # Calculate statistics
                theft_data = dataset.groupby('States/UTs')['THEFT'].sum()
                total_theft = theft_data.sum()
                avg_theft = theft_data.mean()
                max_theft_state = theft_data.idxmax()
                max_theft_value = theft_data.max()
                min_theft_state = theft_data.idxmin()
                min_theft_value = theft_data.min()
                
                # Calculate year-wise trend
                yearly_theft = dataset.groupby('Year')['THEFT'].sum()
                yearly_trend = yearly_theft.pct_change().mean() * 100
                
                # Calculate district-wise analysis for the state with highest theft
                district_theft = dataset[dataset['States/UTs'] == max_theft_state].groupby('District')['THEFT'].sum()
                top_districts = district_theft.nlargest(5)
                
                # Calculate seasonal patterns (if available)
                monthly_theft = dataset.groupby('Year')['THEFT'].mean()
                
                # Generate detailed analysis text with enhanced insights
                strdata += f'''
                <div class="analysis-header mb-4">
                    <h3 class="text-primary">THEFT Crime Analysis</h3>
                    <div class="row g-4">
                        <div class="col-md-6">
                            <div class="card h-100 shadow-sm">
                                <div class="card-body">
                                    <h5 class="card-title">Overall Statistics</h5>
                                    <ul class="list-unstyled">
                                        <li class="mb-2"><strong>Total Cases:</strong> {total_theft:,}</li>
                                        <li class="mb-2"><strong>Average per State:</strong> {avg_theft:.2f}</li>
                                        <li class="mb-2"><strong>Highest Cases:</strong> {max_theft_state} ({max_theft_value:,})</li>
                                        <li class="mb-2"><strong>Lowest Cases:</strong> {min_theft_state} ({min_theft_value:,})</li>
                                        <li class="mb-2"><strong>Yearly Growth:</strong> {yearly_trend:.2f}%</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card h-100 shadow-sm">
                                <div class="card-body">
                                    <h5 class="card-title">Key Insights</h5>
                                    <ul class="list-unstyled">
                                        <li class="mb-2"><strong>Top State Share:</strong> {((max_theft_value/total_theft)*100):.1f}%</li>
                                        <li class="mb-2"><strong>Rate Difference:</strong> {((max_theft_value/min_theft_value)-1)*100:.1f}x</li>
                                        <li class="mb-2"><strong>Above Average States:</strong> {len(theft_data[theft_data > avg_theft])} of {len(theft_data)}</li>
                                        <li class="mb-2"><strong>Crime Hotspots:</strong> Top 5 districts in {max_theft_state}</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                '''
                
                # Generate time series plot
                plt.clf()
                fig = plt.figure(figsize=(15, 7))
                ax = fig.add_subplot(111)
                yearly_theft.plot(kind='line', marker='o', ax=ax)
                plt.title('THEFT Cases Trend Over Years', pad=20)
                plt.xlabel('Year')
                plt.ylabel('Number of THEFT Cases')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig('static/analysis/theft_trend.png', bbox_inches='tight', dpi=100)
                plt.close(fig)
                
                # Generate bar chart with enhanced styling
                plt.clf()
                fig = plt.figure(figsize=(15, 7))
                ax = fig.add_subplot(111)
                bars = theft_data.plot(kind='bar', ax=ax, color='skyblue')
                plt.title('THEFT Cases Distribution Across States', pad=20)
                plt.xlabel('States/UTs')
                plt.ylabel('Number of THEFT Cases')
                plt.xticks(rotation=45, ha='right')
                
                # Add value labels on top of bars
                for bar in bars.patches:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height):,}',
                           ha='center', va='bottom')
                
                plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig('static/analysis/theft_bar.png', bbox_inches='tight', dpi=100)
                plt.close(fig)
                
                # Generate pie chart with enhanced styling
                plt.clf()
                fig = plt.figure(figsize=(12, 9))
                ax = fig.add_subplot(111)
                colors = plt.cm.Pastel1(np.linspace(0, 1, len(theft_data)))
                theft_data.plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=colors)
                plt.title('THEFT Cases Distribution by State (Percentage)', pad=20)
                plt.ylabel('')
                plt.tight_layout()
                plt.savefig('static/analysis/theft_pie.png', bbox_inches='tight', dpi=100)
                plt.close(fig)
                
                # Generate district-wise analysis for top state
                plt.clf()
                fig = plt.figure(figsize=(12, 6))
                ax = fig.add_subplot(111)
                top_districts.plot(kind='barh', ax=ax, color='lightcoral')
                plt.title(f'Top 5 Districts with Highest THEFT Cases in {max_theft_state}', pad=20)
                plt.xlabel('Number of THEFT Cases')
                plt.ylabel('District')
                
                # Add value labels
                for i, v in enumerate(top_districts):
                    ax.text(v, i, f' {int(v):,}', va='center')
                
                plt.grid(True, axis='x', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig('static/analysis/theft_districts.png', bbox_inches='tight', dpi=100)
                plt.close(fig)
                
                # Add graph descriptions with enhanced styling
                strdata += '''
                <div class="row g-4">
                    <div class="col-12">
                        <div class="card shadow-sm">
                            <div class="card-body">
                                <h5 class="card-title">Temporal Analysis</h5>
                                <img src="static/analysis/theft_trend.png" class="img-fluid w-100" alt="THEFT Trend Chart"/>
                                <p class="mt-3 text-muted">Year-wise trend of THEFT cases showing the evolution of crime patterns over time</p>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row g-4 mt-2">
                    <div class="col-md-6">
                        <div class="card shadow-sm">
                            <div class="card-body">
                                <h5 class="card-title">State-wise Distribution</h5>
                                <img src="static/analysis/theft_bar.png" class="img-fluid w-100" alt="THEFT Bar Chart"/>
                                <p class="mt-3 text-muted">Absolute number of THEFT cases across states with detailed statistics</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card shadow-sm">
                            <div class="card-body">
                                <h5 class="card-title">Percentage Distribution</h5>
                                <img src="static/analysis/theft_pie.png" class="img-fluid w-100" alt="THEFT Pie Chart"/>
                                <p class="mt-3 text-muted">Relative proportion of THEFT cases by state showing regional patterns</p>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row g-4 mt-2">
                    <div class="col-12">
                        <div class="card shadow-sm">
                            <div class="card-body">
                                <h5 class="card-title">District-wise Analysis</h5>
                                <img src="static/analysis/theft_districts.png" class="img-fluid w-100" alt="THEFT District Chart"/>
                                <p class="mt-3 text-muted">Detailed breakdown of THEFT cases in the most affected state's districts</p>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row g-4 mt-2">
                    <div class="col-12">
                        <div class="card shadow-sm">
                            <div class="card-body">
                                <h5 class="card-title">Key Observations</h5>
                                <ul class="list-unstyled">
                                    <li class="mb-2">• The state of {max_theft_state} shows significantly higher THEFT cases compared to the national average</li>
                                    <li class="mb-2">• The yearly trend indicates a {yearly_trend > 0 and "growing" or "declining"} pattern in THEFT cases</li>
                                    <li class="mb-2">• Top 5 districts in {max_theft_state} account for a significant portion of total THEFT cases</li>
                                    <li class="mb-2">• There is a substantial variation in THEFT rates between states, suggesting regional factors at play</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
                '''
                
            if classify_type == "Murder":
                # Calculate statistics
                murder_data = dataset.groupby('States/UTs')['Murder'].sum()
                total_murder = murder_data.sum()
                avg_murder = murder_data.mean()
                max_murder_state = murder_data.idxmax()
                max_murder_value = murder_data.max()
                min_murder_state = murder_data.idxmin()
                min_murder_value = murder_data.min()
                
                # Calculate year-wise trend
                yearly_murder = dataset.groupby('Year')['Murder'].sum()
                yearly_trend = yearly_murder.pct_change().mean() * 100
                
                # Calculate district-wise analysis for the state with highest murder
                district_murder = dataset[dataset['States/UTs'] == max_murder_state].groupby('District')['Murder'].sum()
                top_districts = district_murder.nlargest(5)
                
                # Calculate seasonal patterns (if available)
                monthly_murder = dataset.groupby('Year')['Murder'].mean()
                
                # Generate detailed analysis text with enhanced insights
                strdata += f'''
                <div class="analysis-header mb-4">
                    <h3 class="text-primary">Murder Crime Analysis</h3>
                    <div class="row g-4">
                        <div class="col-md-6">
                            <div class="card h-100 shadow-sm">
                                <div class="card-body">
                                    <h5 class="card-title">Overall Statistics</h5>
                                    <ul class="list-unstyled">
                                        <li class="mb-2"><strong>Total Cases:</strong> {total_murder:,}</li>
                                        <li class="mb-2"><strong>Average per State:</strong> {avg_murder:.2f}</li>
                                        <li class="mb-2"><strong>Highest Cases:</strong> {max_murder_state} ({max_murder_value:,})</li>
                                        <li class="mb-2"><strong>Lowest Cases:</strong> {min_murder_state} ({min_murder_value:,})</li>
                                        <li class="mb-2"><strong>Yearly Growth:</strong> {yearly_trend:.2f}%</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card h-100 shadow-sm">
                                <div class="card-body">
                                    <h5 class="card-title">Key Insights</h5>
                                    <ul class="list-unstyled">
                                        <li class="mb-2"><strong>Top State Share:</strong> {((max_murder_value/total_murder)*100):.1f}%</li>
                                        <li class="mb-2"><strong>Rate Difference:</strong> {((max_murder_value/min_murder_value)-1)*100:.1f}x</li>
                                        <li class="mb-2"><strong>Above Average States:</strong> {len(murder_data[murder_data > avg_murder])} of {len(murder_data)}</li>
                                        <li class="mb-2"><strong>Crime Hotspots:</strong> Top 5 districts in {max_murder_state}</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                '''
                
                # Generate time series plot
                plt.clf()
                fig = plt.figure(figsize=(15, 7))
                ax = fig.add_subplot(111)
                yearly_murder.plot(kind='line', marker='o', ax=ax)
                plt.title('Murder Cases Trend Over Years', pad=20)
                plt.xlabel('Year')
                plt.ylabel('Number of Murder Cases')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig('static/analysis/murder_trend.png', bbox_inches='tight', dpi=100)
                plt.close(fig)
                
                # Generate bar chart with enhanced styling
                plt.clf()
                fig = plt.figure(figsize=(15, 7))
                ax = fig.add_subplot(111)
                bars = murder_data.plot(kind='bar', ax=ax, color='salmon')
                plt.title('Murder Cases Distribution Across States', pad=20)
                plt.xlabel('States/UTs')
                plt.ylabel('Number of Murder Cases')
                plt.xticks(rotation=45, ha='right')
                
                # Add value labels on top of bars
                for bar in bars.patches:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height):,}',
                           ha='center', va='bottom')
                
                plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig('static/analysis/murder_bar.png', bbox_inches='tight', dpi=100)
                plt.close(fig)
                
                # Generate pie chart with enhanced styling
                plt.clf()
                fig = plt.figure(figsize=(12, 9))
                ax = fig.add_subplot(111)
                colors = plt.cm.Pastel1(np.linspace(0, 1, len(murder_data)))
                murder_data.plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=colors)
                plt.title('Murder Cases Distribution by State (Percentage)', pad=20)
                plt.ylabel('')
                plt.tight_layout()
                plt.savefig('static/analysis/murder_pie.png', bbox_inches='tight', dpi=100)
                plt.close(fig)
                
                # Generate district-wise analysis for top state
                plt.clf()
                fig = plt.figure(figsize=(12, 6))
                ax = fig.add_subplot(111)
                top_districts.plot(kind='barh', ax=ax, color='indianred')
                plt.title(f'Top 5 Districts with Highest Murder Cases in {max_murder_state}', pad=20)
                plt.xlabel('Number of Murder Cases')
                plt.ylabel('District')
                
                # Add value labels
                for i, v in enumerate(top_districts):
                    ax.text(v, i, f' {int(v):,}', va='center')
                
                plt.grid(True, axis='x', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig('static/analysis/murder_districts.png', bbox_inches='tight', dpi=100)
                plt.close(fig)
                
                # Add graph descriptions with enhanced styling
                strdata += '''
                <div class="row g-4">
                    <div class="col-12">
                        <div class="card shadow-sm">
                            <div class="card-body">
                                <h5 class="card-title">Temporal Analysis</h5>
                                <img src="static/analysis/murder_trend.png" class="img-fluid w-100" alt="Murder Trend Chart"/>
                                <p class="mt-3 text-muted">Year-wise trend of Murder cases showing the evolution of crime patterns over time</p>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row g-4 mt-2">
                    <div class="col-md-6">
                        <div class="card shadow-sm">
                            <div class="card-body">
                                <h5 class="card-title">State-wise Distribution</h5>
                                <img src="static/analysis/murder_bar.png" class="img-fluid w-100" alt="Murder Bar Chart"/>
                                <p class="mt-3 text-muted">Absolute number of Murder cases across states with detailed statistics</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card shadow-sm">
                            <div class="card-body">
                                <h5 class="card-title">Percentage Distribution</h5>
                                <img src="static/analysis/murder_pie.png" class="img-fluid w-100" alt="Murder Pie Chart"/>
                                <p class="mt-3 text-muted">Relative proportion of Murder cases by state showing regional patterns</p>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row g-4 mt-2">
                    <div class="col-12">
                        <div class="card shadow-sm">
                            <div class="card-body">
                                <h5 class="card-title">District-wise Analysis</h5>
                                <img src="static/analysis/murder_districts.png" class="img-fluid w-100" alt="Murder District Chart"/>
                                <p class="mt-3 text-muted">Detailed breakdown of Murder cases in the most affected state's districts</p>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row g-4 mt-2">
                    <div class="col-12">
                        <div class="card shadow-sm">
                            <div class="card-body">
                                <h5 class="card-title">Key Observations</h5>
                                <ul class="list-unstyled">
                                    <li class="mb-2">• The state of {max_murder_state} shows significantly higher Murder cases compared to the national average</li>
                                    <li class="mb-2">• The yearly trend indicates a {yearly_trend > 0 and "growing" or "declining"} pattern in Murder cases</li>
                                    <li class="mb-2">• Top 5 districts in {max_murder_state} account for a significant portion of total Murder cases</li>
                                    <li class="mb-2">• There is a substantial variation in Murder rates between states, suggesting regional factors at play</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
                '''
                
            if classify_type == "Rape":
                # Calculate statistics
                rape_data = dataset.groupby('States/UTs')['RAPE'].sum()
                total_rape = rape_data.sum()
                avg_rape = rape_data.mean()
                max_rape_state = rape_data.idxmax()
                max_rape_value = rape_data.max()
                min_rape_state = rape_data.idxmin()
                min_rape_value = rape_data.min()
                
                # Calculate year-wise trend
                yearly_rape = dataset.groupby('Year')['RAPE'].sum()
                yearly_trend = yearly_rape.pct_change().mean() * 100
                
                # Calculate district-wise analysis for the state with highest rape
                district_rape = dataset[dataset['States/UTs'] == max_rape_state].groupby('District')['RAPE'].sum()
                top_districts = district_rape.nlargest(5)
                
                # Calculate seasonal patterns (if available)
                monthly_rape = dataset.groupby('Year')['RAPE'].mean()
                
                # Generate detailed analysis text with enhanced insights
                strdata += f'''
                <div class="analysis-header mb-4">
                    <h3 class="text-primary">Rape Crime Analysis</h3>
                    <div class="row g-4">
                        <div class="col-md-6">
                            <div class="card h-100 shadow-sm">
                                <div class="card-body">
                                    <h5 class="card-title">Overall Statistics</h5>
                                    <ul class="list-unstyled">
                                        <li class="mb-2"><strong>Total Cases:</strong> {total_rape:,}</li>
                                        <li class="mb-2"><strong>Average per State:</strong> {avg_rape:.2f}</li>
                                        <li class="mb-2"><strong>Highest Cases:</strong> {max_rape_state} ({max_rape_value:,})</li>
                                        <li class="mb-2"><strong>Lowest Cases:</strong> {min_rape_state} ({min_rape_value:,})</li>
                                        <li class="mb-2"><strong>Yearly Growth:</strong> {yearly_trend:.2f}%</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card h-100 shadow-sm">
                                <div class="card-body">
                                    <h5 class="card-title">Key Insights</h5>
                                    <ul class="list-unstyled">
                                        <li class="mb-2"><strong>Top State Share:</strong> {((max_rape_value/total_rape)*100):.1f}%</li>
                                        <li class="mb-2"><strong>Rate Difference:</strong> {((max_rape_value/min_rape_value)-1)*100:.1f}x</li>
                                        <li class="mb-2"><strong>Above Average States:</strong> {len(rape_data[rape_data > avg_rape])} of {len(rape_data)}</li>
                                        <li class="mb-2"><strong>Crime Hotspots:</strong> Top 5 districts in {max_rape_state}</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                '''
                
                # Generate time series plot
                plt.clf()
                fig = plt.figure(figsize=(15, 7))
                ax = fig.add_subplot(111)
                yearly_rape.plot(kind='line', marker='o', ax=ax)
                plt.title('Rape Cases Trend Over Years', pad=20)
                plt.xlabel('Year')
                plt.ylabel('Number of Rape Cases')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig('static/analysis/rape_trend.png', bbox_inches='tight', dpi=100)
                plt.close(fig)
                
                # Generate bar chart with enhanced styling
                plt.clf()
                fig = plt.figure(figsize=(15, 7))
                ax = fig.add_subplot(111)
                bars = rape_data.plot(kind='bar', ax=ax, color='plum')
                plt.title('Rape Cases Distribution Across States', pad=20)
                plt.xlabel('States/UTs')
                plt.ylabel('Number of Rape Cases')
                plt.xticks(rotation=45, ha='right')
                
                # Add value labels on top of bars
                for bar in bars.patches:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height):,}',
                           ha='center', va='bottom')
                
                plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig('static/analysis/rape_bar.png', bbox_inches='tight', dpi=100)
                plt.close(fig)
                
                # Generate pie chart with enhanced styling
                plt.clf()
                fig = plt.figure(figsize=(12, 9))
                ax = fig.add_subplot(111)
                colors = plt.cm.Pastel1(np.linspace(0, 1, len(rape_data)))
                rape_data.plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=colors)
                plt.title('Rape Cases Distribution by State (Percentage)', pad=20)
                plt.ylabel('')
                plt.tight_layout()
                plt.savefig('static/analysis/rape_pie.png', bbox_inches='tight', dpi=100)
                plt.close(fig)
                
                # Generate district-wise analysis for top state
                plt.clf()
                fig = plt.figure(figsize=(12, 6))
                ax = fig.add_subplot(111)
                top_districts.plot(kind='barh', ax=ax, color='mediumpurple')
                plt.title(f'Top 5 Districts with Highest Rape Cases in {max_rape_state}', pad=20)
                plt.xlabel('Number of Rape Cases')
                plt.ylabel('District')
                
                # Add value labels
                for i, v in enumerate(top_districts):
                    ax.text(v, i, f' {int(v):,}', va='center')
                
                plt.grid(True, axis='x', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig('static/analysis/rape_districts.png', bbox_inches='tight', dpi=100)
                plt.close(fig)
                
                # Add graph descriptions with enhanced styling
                strdata += '''
                <div class="row g-4">
                    <div class="col-12">
                        <div class="card shadow-sm">
                            <div class="card-body">
                                <h5 class="card-title">Temporal Analysis</h5>
                                <img src="static/analysis/rape_trend.png" class="img-fluid w-100" alt="Rape Trend Chart"/>
                                <p class="mt-3 text-muted">Year-wise trend of Rape cases showing the evolution of crime patterns over time</p>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row g-4 mt-2">
                    <div class="col-md-6">
                        <div class="card shadow-sm">
                            <div class="card-body">
                                <h5 class="card-title">State-wise Distribution</h5>
                                <img src="static/analysis/rape_bar.png" class="img-fluid w-100" alt="Rape Bar Chart"/>
                                <p class="mt-3 text-muted">Absolute number of Rape cases across states with detailed statistics</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card shadow-sm">
                            <div class="card-body">
                                <h5 class="card-title">Percentage Distribution</h5>
                                <img src="static/analysis/rape_pie.png" class="img-fluid w-100" alt="Rape Pie Chart"/>
                                <p class="mt-3 text-muted">Relative proportion of Rape cases by state showing regional patterns</p>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row g-4 mt-2">
                    <div class="col-12">
                        <div class="card shadow-sm">
                            <div class="card-body">
                                <h5 class="card-title">District-wise Analysis</h5>
                                <img src="static/analysis/rape_districts.png" class="img-fluid w-100" alt="Rape District Chart"/>
                                <p class="mt-3 text-muted">Detailed breakdown of Rape cases in the most affected state's districts</p>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row g-4 mt-2">
                    <div class="col-12">
                        <div class="card shadow-sm">
                            <div class="card-body">
                                <h5 class="card-title">Key Observations</h5>
                                <ul class="list-unstyled">
                                    <li class="mb-2">• The state of {max_rape_state} shows significantly higher Rape cases compared to the national average</li>
                                    <li class="mb-2">• The yearly trend indicates a {yearly_trend > 0 and "growing" or "declining"} pattern in Rape cases</li>
                                    <li class="mb-2">• Top 5 districts in {max_rape_state} account for a significant portion of total Rape cases</li>
                                    <li class="mb-2">• There is a substantial variation in Rape rates between states, suggesting regional factors at play</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
                '''
                
        except Exception as e:
            strdata += f'''
            <div class="alert alert-danger">
                <h4>Error in Analysis</h4>
                <p>An error occurred while generating the analysis: {str(e)}</p>
                <pre class="mt-2">{traceback.format_exc()}</pre>
            </div>
            '''
            
        strdata += '''
                </div>
            </div>
        </div>
        '''
            
        context = {'data': strdata}            
        return render(request, 'ViewGraphs.html', context)
        
def ViewDataset(request):
    if request.method == 'GET':
        dataset = pd.read_csv("Dataset/cleaned_crime_data.csv")
        strdata = '<table border=1 align=center width=100%><tr><th><font size="" color="black">States/UTs</th><th><font size="" color="black">District</th><th><font size="" color="black">Year</th><th><font size="" color="black">THEFT</th><th><font size="" color="black">Murder</th><th><font size="" color="black">RAPE</th></tr>'
        for i in range(len(dataset)):
            strdata += '<tr><td><font size="" color="black">'+str(dataset.iloc[i,0])+'</td><td><font size="" color="black">'+str(dataset.iloc[i,1])+'</td><td><font size="" color="black">'+str(dataset.iloc[i,2])+'</td><td><font size="" color="black">'+str(dataset.iloc[i,3])+'</td><td><font size="" color="black">'+str(dataset.iloc[i,4])+'</td><td><font size="" color="black">'+str(dataset.iloc[i,5])+'</td></tr>'
        strdata += '</table>'
        context = {'data': strdata}
        return render(request, 'ViewDataset.html', context)

def get_districts_by_state(request):
    if request.method == 'GET':
        state = request.GET.get('state', '')
        if state:
            dataset = pd.read_csv("Dataset/cleaned_crime_data.csv", usecols=['States/UTs','District'])
            dataset.fillna(0, inplace = True)
            
            # Filter districts for the selected state
            state_districts = dataset[dataset['States/UTs'] == state]['District'].unique()
            
            # Create HTML options
            districts_html = '<option value="">Select District</option>'
            for district in state_districts:
                districts_html += f'<option value="{district}">{district}</option>'
            
            return HttpResponse(districts_html)
    return HttpResponse('')





    
    
