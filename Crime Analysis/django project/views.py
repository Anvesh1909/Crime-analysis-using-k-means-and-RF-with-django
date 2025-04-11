def generate_key_observations(crime_type, data):
    observations = []
    
    if crime_type == 'Theft':
        # Calculate key statistics
        total_thefts = data['Total_Thefts'].sum()
        avg_thefts = data['Total_Thefts'].mean()
        max_state = data.loc[data['Total_Thefts'].idxmax(), 'State/UT']
        max_thefts = data['Total_Thefts'].max()
        
        observations = [
            {
                'title': 'Overall Theft Statistics',
                'description': f'India recorded a total of {total_thefts:,.0f} theft cases. The average number of thefts per state/UT is {avg_thefts:,.0f}, indicating significant variation across regions.'
            },
            {
                'title': 'Regional Analysis',
                'description': f'{max_state} reported the highest number of theft cases with {max_thefts:,.0f} incidents, suggesting potential areas for targeted intervention.'
            },
            {
                'title': 'Trend Analysis',
                'description': 'The data shows varying patterns across states, with some regions showing higher vulnerability to theft-related crimes.'
            }
        ]
    
    elif crime_type == 'Rape':
        # Calculate key statistics
        total_rapes = data['Total_Rapes'].sum()
        avg_rapes = data['Total_Rapes'].mean()
        max_state = data.loc[data['Total_Rapes'].idxmax(), 'State/UT']
        max_rapes = data['Total_Rapes'].max()
        
        observations = [
            {
                'title': 'Overall Rape Statistics',
                'description': f'India recorded {total_rapes:,.0f} reported rape cases. The average number of cases per state/UT is {avg_rapes:,.0f}, highlighting the severity of this issue.'
            },
            {
                'title': 'Regional Analysis',
                'description': f'{max_state} reported the highest number of rape cases with {max_rapes:,.0f} incidents, indicating a critical need for enhanced security measures and awareness programs.'
            },
            {
                'title': 'Trend Analysis',
                'description': 'The data reveals concerning patterns in certain regions, suggesting the need for targeted intervention and policy changes.'
            }
        ]
    
    return observations

def generate_graph_descriptions(crime_type, graph_type):
    descriptions = {
        'Theft': {
            'state_distribution': 'This graph shows the distribution of theft cases across different states and union territories. The height of each bar represents the total number of reported theft cases in that region.',
            'monthly_trend': 'This visualization displays the monthly trend of theft cases throughout the year. It helps identify any seasonal patterns or periods of increased criminal activity.',
            'category_breakdown': 'This chart breaks down theft cases by different categories, providing insights into the types of thefts most commonly reported.',
            'comparison': 'This comparative analysis shows the relationship between different types of theft cases, helping identify correlations and patterns.'
        },
        'Rape': {
            'state_distribution': 'This graph illustrates the distribution of reported rape cases across different states and union territories. The height of each bar represents the total number of cases in that region.',
            'monthly_trend': 'This visualization shows the monthly trend of reported rape cases throughout the year, helping identify any temporal patterns or seasonal variations.',
            'category_breakdown': 'This chart provides a detailed breakdown of rape cases by different categories, offering insights into the nature and circumstances of reported incidents.',
            'comparison': 'This comparative analysis examines the relationship between different aspects of rape cases, helping identify patterns and potential areas for intervention.'
        }
    }
    
    return descriptions.get(crime_type, {}).get(graph_type, 'Detailed analysis of crime data.') 