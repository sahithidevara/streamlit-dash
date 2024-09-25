import pandas as pd
import streamlit as st
import plotly.express as px


def create_hourly_line_chart(data):
    
    fig = px.line(
        data,
        x='Post_Hour',
        y='Count',
        # title=f"Hourly Post Counts",
        labels={'Post_Hour': 'Hour of Day', 'Count': 'Number of Posts'},
        markers=True
    )
    
    fig.update_traces(
        mode='lines+markers+text',
        text=data['Count'],
        textposition='top center',
        marker=dict(size=8, color='red')  # Customize marker appearance
    )
    
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(0, 24)),
            ticktext=[str(i) for i in range(0, 24)],
            title='Hour of Day'
        )
    )
    
    return fig

def create_pie_chart(data,text):
    # Define a color mapping for each sentiment type
    color_map = {
        'Positive': 'green',
        'Negative': 'red',
        'Neutral': 'yellow'
    }

    # Create pie charts with consistent colors
    fig = px.pie(
        data,
        names='Sentiment_Type',
        values='Count',
        title=f"Sentiment Distribution by {text}",
        hole=0.3,  # Optional: Adds a hole in the middle for a donut chart effect
        color='Sentiment_Type',  # Use the Sentiment_Type column for colors
        color_discrete_map=color_map  # Ensure consistent colors
    )

    # Add percentage labels inside the pie charts
    fig.update_traces(
        textinfo='label+percent',
        textfont_size=14
    )

    return fig

def generate_comparison_columns(filtered_raw_data, filtered_raw_data_after_removal, Type_data):
    col1, col2,col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("#### Average Sentiment Score Comparison")
        # Calculate the mean sentiment score before and after removal
        original_sentiment_score = round(filtered_raw_data['Sentiment_Score'].mean(), 4)
        after_removal_Negative_Authors_sentiment_score = round(filtered_raw_data_after_removal['Sentiment_Score'].mean(), 4)
        sentiment_scores = pd.DataFrame({
            'Category': ['Raw Data', 'After Removal of -Ve Authors'],
            'Sentiment Score': [original_sentiment_score, after_removal_Negative_Authors_sentiment_score]
        })
        tab1, tab2 = st.tabs(["📶 Chart", "☛ Data"])
        with tab1:
            # Create a bar chart with Plotly, using a heatmap-like color scale
            fig = px.bar(sentiment_scores, x='Category', y='Sentiment Score', text='Sentiment Score',color='Sentiment Score', color_continuous_scale='RdYlGn')
            # Update layout to show colorbar and adjust the display
            fig.update_layout(coloraxis_showscale=False)

            # Display the chart in Streamlit
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            st.dataframe(sentiment_scores,use_container_width=True)


    with col2: 
        st.markdown("#### Average Replies to Posts Comparison")
        # Calculate the mean sentiment score before and after removal
        original_sentiment_score = round(Type_data['Reply']/Type_data['Post'], 3)
        after_removal_type_data = filtered_raw_data_after_removal['Type'].value_counts()
        after_removal_Negative_Authors_sentiment_score = round(after_removal_type_data['Reply']/after_removal_type_data['Post'], 3)

        # Create a DataFrame to hold both scores
        Avg_replies_to_posts = pd.DataFrame({
            'Category': ['Raw Data', 'After Removal of -Ve Authors'],
            'Average Replies to Posts': [original_sentiment_score, after_removal_Negative_Authors_sentiment_score]
        })

        tab1, tab2 = st.tabs(["📶 Chart", "☛ Data"])
        with tab1:
        # Create a bar chart with Plotly
            fig = px.bar(Avg_replies_to_posts, x='Category', y='Average Replies to Posts', text='Average Replies to Posts',color='Average Replies to Posts', color_continuous_scale='RdYlGn')
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            st.dataframe(Avg_replies_to_posts,use_container_width=True)

    with col3:
        st.markdown("#### Average Score Comparison")
        # Calculate the mean sentiment score before and after removal
        original_score = round(filtered_raw_data['Score'].mean(), 4)
        after_removal_Negative_Authors_score = round(filtered_raw_data_after_removal['Score'].mean(), 4)
        scores = pd.DataFrame({
            'Category': ['Raw Data', 'After Removal of -Ve Authors'],
            'Score': [original_score, after_removal_Negative_Authors_score]
        })
        tab1, tab2 = st.tabs(["📶 Chart", "☛ Data"])
        with tab1:
            # Create a bar chart with Plotly, using a heatmap-like color scale
            fig = px.bar(scores, x='Category', y='Score', text='Score',color='Score', color_continuous_scale='RdYlGn')
            # Update layout to show colorbar and adjust the display
            fig.update_layout(coloraxis_showscale=False)

            # Display the chart in Streamlit
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            st.dataframe(scores,use_container_width=True)

# Set page configuration
st.set_page_config(
    page_title="Reddit Sentimentail Analysis",
    page_icon=":rocket:",  # This can be an emoji or path to an image
    layout="wide",  # Options: "centered" or "wide"
    initial_sidebar_state="auto",  # Options: "auto", "expanded", "collapsed"
)

# Your app code starts here
st.title(":bar_chart: Sentimental Analysis on Reddit Communities")
# st.markdown('#')
st.sidebar.title("Filters")

@st.cache_data
def collect_data(filename,sheetname):
    data = pd.read_excel(filename,sheet_name=sheetname)
    return data

merged_raw_data = collect_data("data/Combined_Analysissheet.xlsx","raw_data")
merged_raw_data['Post_DateTime'] = pd.to_datetime(merged_raw_data['Post_DateTime'])
merged_unique_authors_data = collect_data("data/Combined_Analysissheet.xlsx",'unique_authors')

communities = merged_raw_data['Community'].value_counts()
# st.dataframe(communities)
unique_authors = merged_unique_authors_data['Community'].value_counts()
# st.dataframe(unique_authors)
# Display unique communities and their counts as bullet points
st.markdown("## List of Communities with Analysed submissions")
for community, count in communities.items():
    authors_count = unique_authors.get(community, 0)  # Get the number of unique authors or 0 if community not in unique_authors
    st.markdown(f"- #### {community} :  <span style='color:#f75402'><b>{count}</b></span> submissions by <span style='color:#5c05e8'><b>{authors_count}</b></span> authors", unsafe_allow_html=True)

selected_community = st.sidebar.selectbox("Select Community", options=communities.keys())
# Filter the dataframe based on the selected community
#raw data
filtered_raw_data = merged_raw_data[merged_raw_data['Community'] == selected_community]
filtered_raw_data = filtered_raw_data.reset_index(drop=True)
# st.dataframe(filtered_raw_data)
filtered_unique_authors_data = merged_unique_authors_data[merged_unique_authors_data['Community'] == selected_community]
filtered_unique_authors_data = filtered_unique_authors_data.reset_index(drop=True)
st.markdown(f"# Selected Community: <span style='color:#f70202'><b>{selected_community}</b></span> ", unsafe_allow_html=True)

# Total posts and replies to posts
Type_data = filtered_raw_data['Type'].value_counts()
# st.dataframe(Type_data)
st.markdown(f"### Posts: <span style='color:#f70202'><b>{Type_data['Post']}</b></span>",unsafe_allow_html=True)
st.markdown(f"### Replies to the posts:  <span style='color:#f70202'><b>{Type_data['Reply']}</b></span>",unsafe_allow_html=True)
st.markdown(f"### Average Like Score:  <span style='color:#f70202'><b>{round(filtered_raw_data['Score'].mean(),4)}</b></span>",unsafe_allow_html=True)
st.markdown(f"### Average Sentiment Score:  <span style='color:#f70202'><b>{round(filtered_raw_data['Sentiment_Score'].mean(),4)}</b></span>",unsafe_allow_html=True)
st.markdown(f"### Average Replies to Posts: <span style='color:#f70202'><b>{round(Type_data['Reply']/Type_data['Post'], 3)}</b></span>",unsafe_allow_html=True)


filtered_raw_data['Post_Hour'] = filtered_raw_data['Post_DateTime'].dt.hour
filtered_raw_data['Day_of_Week'] = filtered_raw_data['Post_DateTime'].dt.day_name()
# st.dataframe(filtered_raw_data)
# filtered_raw_data['Post_Day'] = filtered_raw_data['Post_DateTime'].dt.day
# filtered_raw_data['Post_Month'] = filtered_raw_data['Post_DateTime'].dt.month
# filtered_raw_data['Post_Year'] = filtered_raw_data['Post_DateTime'].dt.year

# HOUR IN DAY
Hour_data = filtered_raw_data.groupby('Post_Hour').size().reset_index(name='Count')
# st.dataframe(Hour_data)
# DAY OF WEEK
Week_day_data = filtered_raw_data.groupby('Day_of_Week').size().reset_index(name='Count')
# st.dataframe(Week_day_data)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
Week_day_data['Day_of_Week'] = pd.Categorical(Week_day_data['Day_of_Week'], categories=day_order, ordered=True)
Week_day_data = Week_day_data.sort_values('Day_of_Week')

st.markdown('## @ Posts activity')
# Create a bar chart
fig_Week_day_data_bar = px.bar(
    Week_day_data,
    x='Day_of_Week',
    y='Count',
    title=f"Posts by Day of Week in {selected_community}",
    labels={'Day_of_Week': 'Day of Week', 'Count': 'Number of Posts'},
    color='Count',  # Optional: Color by count for better visualization
    color_continuous_scale='Viridis'  # Optional: Use a color scale
)

fig_Week_day_data_bar.update_traces(
    text=Week_day_data['Count'],
    # textposition='outside',  # Positions text outside the bars
    texttemplate='%{text}'  # Displays text as it is
)

# Display the bar chart
# Display the pie charts side by side
tab1, tab2 = st.tabs(["By Week", "By Day"])

with tab1:
    st.plotly_chart(fig_Week_day_data_bar, use_container_width=True)

with tab2:
    # Hour line bar
    fig_positive = create_hourly_line_chart(Hour_data)
    st.plotly_chart(fig_positive, use_container_width=True)

sentiment_order = ['Positive', 'Negative', 'Neutral']
Sentiment_Type_data = filtered_raw_data.groupby('Sentiment_Type').size().reset_index(name='Count')
Sentiment_Type_data['Sentiment_Type'] = pd.Categorical(Sentiment_Type_data['Sentiment_Type'], categories=sentiment_order, ordered=True)
Sentiment_Type_data = Sentiment_Type_data.sort_values('Sentiment_Type')
# st.dataframe(Sentiment_Type_data)

Sentiment_Type_unique_authors_data = filtered_unique_authors_data.groupby('Sentiment_Type').size().reset_index(name='Count')
Sentiment_Type_unique_authors_data['Sentiment_Type'] = pd.Categorical(Sentiment_Type_unique_authors_data['Sentiment_Type'], categories=sentiment_order, ordered=True)
Sentiment_Type_unique_authors_data = Sentiment_Type_unique_authors_data.sort_values('Sentiment_Type')
# st.dataframe(Sentiment_Type_unique_authors_data)

# Display the pie charts side by side
col1, col2 = st.columns(2)

with col1:
    fig_pie_posts = create_pie_chart(Sentiment_Type_data,'Posts')
    st.plotly_chart(fig_pie_posts, use_container_width=True)

with col2:
    fig_pie_authors = create_pie_chart(Sentiment_Type_unique_authors_data,'Unique Authors')
    st.plotly_chart(fig_pie_authors, use_container_width=True)

# Group by Sentiment_Type and Post_Hour and count the number of posts
sentimentType_posthour_data = filtered_raw_data.groupby(['Sentiment_Type', 'Post_Hour']).size().reset_index(name='Count')

# Create tabs for each sentiment type
st.markdown("### @ Hourly Post Counts based on Sentiment Type")
tab1, tab2 = st.tabs(["📈 Line Chart","☛ Top 5 Activity hours"])

with tab1:
    fig = px.line(sentimentType_posthour_data, 
                x='Post_Hour', 
                y='Count', 
                color='Sentiment_Type',
                title='Hourly Post Count by Sentiment Type',
                labels={'Post_Hour': 'Hour of the Day', 'Count': 'Number of Posts'},
                markers=True)
    fig.update_traces(textposition='top center')
        
    fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(0, 24)),
                ticktext=[str(i) for i in range(0, 24)],
                title='Hour of Day'
            )
        )
    st.plotly_chart(fig, use_container_width=True)


with tab2:
    col1, col2, col3 = st.columns([1,1, 1])
    with col1: 
        positive_sentiment_data = sentimentType_posthour_data[sentimentType_posthour_data['Sentiment_Type'] == 'Positive']
        st.markdown("#### Positive Sentiment submissions")
        st.markdown("######")
        st.dataframe(positive_sentiment_data.sort_values(by='Count', ascending=False).head(5).reset_index(drop=True).drop(columns=['Sentiment_Type']),use_container_width=True)

    # In the second column, display the dataframe
    with col2:
        negative_sentiment_data = sentimentType_posthour_data[sentimentType_posthour_data['Sentiment_Type'] == 'Negative']
        st.markdown("#### Negative Sentiment submissions")
        st.markdown("######")
        st.dataframe(negative_sentiment_data.sort_values(by='Count', ascending=False).head(5).reset_index(drop=True).drop(columns=['Sentiment_Type']),use_container_width=True)

    with col3:
        neutral_sentiment_data = sentimentType_posthour_data[sentimentType_posthour_data['Sentiment_Type'] == 'Neutral']
        st.markdown("#### Neutral Sentiment submissions")
        st.markdown("######")
        st.dataframe(neutral_sentiment_data.sort_values(by='Count', ascending=False).head(5).reset_index(drop=True).drop(columns=['Sentiment_Type']),use_container_width=True)

# with tab2:
#     st.markdown("#### Negative Sentiment")
#     col1, col2 = st.columns([3, 1])
#     with col1: 
#         negative_sentiment_data = sentimentType_posthour_data[sentimentType_posthour_data['Sentiment_Type'] == 'Negative']
#         fig_negative = create_hourly_line_chart(negative_sentiment_data)
#         st.plotly_chart(fig_negative, use_container_width=True)

#     # In the second column, display the dataframe
#     with col2:
#         st.markdown("#### Top 5 Activity hour with Negative Sentiment submissions")
#         st.markdown("######")
#         st.dataframe(negative_sentiment_data.sort_values(by='Count', ascending=False).head(5).reset_index(drop=True).drop(columns=['Sentiment_Type']),use_container_width=True)

# with tab3:
#     st.markdown("#### Neutral Sentiment")
#     col1, col2 = st.columns([3, 1])
#     with col1: 
#         neutral_sentiment_data = sentimentType_posthour_data[sentimentType_posthour_data['Sentiment_Type'] == 'Neutral']
#         fig_neutral = create_hourly_line_chart(neutral_sentiment_data)
#         st.plotly_chart(fig_neutral, use_container_width=True)

#     # In the second column, display the dataframe
#     with col2:
#         st.markdown("#### Top 5 Activity hour with Neutral Sentiment submissions")
#         st.markdown("######")
#         st.dataframe(neutral_sentiment_data.sort_values(by='Count', ascending=False).head(5).reset_index(drop=True).drop(columns=['Sentiment_Type']),use_container_width=True)

st.markdown("### @ Top 10 Authors by Submission Counts")
top_25_authors = filtered_unique_authors_data.sort_values(by='Submission_Count', ascending=False).head(10)

# Create the bar chart using Plotly
fig = px.bar(top_25_authors, x='Author', y='Submission_Count', 
             text='Submission_Count')

st.plotly_chart(fig, use_container_width=True)

st.markdown("### @ Negative Post Authors based Sentimental Score")
# Filter for authors with negative sentiment posts
negative_authors = filtered_unique_authors_data[(filtered_unique_authors_data['Sentiment_Type'] == 'Negative') & (filtered_unique_authors_data['Post_Count'] != 0)]
post_authors = filtered_unique_authors_data[(filtered_unique_authors_data['Post_Count'] != 0)]
# Sort by submission count or any other relevant column and limit to top 25 authors
top_negative_authors = negative_authors.sort_values(by='Avg_Sentiment_Score', ascending=True)

st.markdown(f"#### Total Negative Authors in {selected_community} : <span style='color:#f70202'><b>{negative_authors.shape[0]}</b></span> out of <span style='color:#f70202'><b>{post_authors.shape[0]}</b></span>",unsafe_allow_html = True)
# Create the bar chart using Plotly, adding text for the values
fig = px.bar(top_negative_authors, x='Author', y='Avg_Sentiment_Score', 
             title='Authors with Most Negative Sentiment',
             text='Submission_Count')

# Display the chart in Streamlit
st.plotly_chart(fig, use_container_width=True)

st.markdown("### @ Remove Negative Post Authors based Sentimental Score")

filtered_reply_data= filtered_raw_data[filtered_raw_data['Type']=='Reply']
# Find the list of Authors to each Target_Author
authors_connections = filtered_reply_data.groupby('Target_Author')['Author'].apply(list).reset_index(name='Author_List')
# Length of Author_List
authors_connections['Author_Count'] = authors_connections['Author_List'].apply(lambda x: len(x))
authors_connections = authors_connections[['Target_Author','Author_Count']]
authors_connections = authors_connections.rename(columns={'Target_Author': 'Author','Author_Count': 'Connections'})
filtered_negative_authors = negative_authors[['Author', 'Post_Count', 'Reply_Count', 'Avg_Sentiment_Score', 'Sentiment_Type']]
filtered_negative_authors = filtered_negative_authors.merge(authors_connections, on='Author', how='left')
filtered_negative_authors['Connections'] = filtered_negative_authors['Connections'].fillna(0)
filtered_negative_authors= filtered_negative_authors.sort_values(by= 'Connections',ascending= True).reset_index(drop = True)

# Display the merged DataFrame
st.dataframe(filtered_negative_authors)
tab1, tab2 = st.tabs(["Select Authors","By Percentage"])
# Percentage Slider
with tab1 : 
    Selected_Negative_Authors = st.multiselect('Select the Author', options=filtered_negative_authors['Author'].unique())
    filtered_negative_posts = filtered_raw_data[filtered_raw_data['Author'].isin(Selected_Negative_Authors)]
    # Get the list of Post_ID and Reply_ID
    negative_authors_post_ids = filtered_negative_posts['Post_ID']
    # Filter the raw data to exclude the Post_IDs from negative authors
    filtered_raw_data_after_removal = filtered_raw_data[~filtered_raw_data['Post_ID'].isin(negative_authors_post_ids)]
    generate_comparison_columns(filtered_raw_data, filtered_raw_data_after_removal, Type_data)

with tab2 : 
    percentage = st.slider('%', 0, 100, step=5)
    remove_negative_count = round((filtered_negative_authors.shape[0]*percentage/100))
    remove_negative_authors= filtered_negative_authors.sort_values(by= 'Connections',ascending= True).reset_index(drop=True).head(remove_negative_count)
    Array_remove_negative_authors = remove_negative_authors['Author']
    #Get the list of Post_ID and Reply_ID by Array_remove_negative_authors from filtered_raw_data
    # Filter the raw data based on the list of negative authors
    filtered_negative_posts = filtered_raw_data[filtered_raw_data['Author'].isin(Array_remove_negative_authors)]
    # Get the list of Post_ID and Reply_ID
    negative_authors_post_ids = filtered_negative_posts['Post_ID']
    # Filter the raw data to exclude the Post_IDs from negative authors
    filtered_raw_data_after_removal = filtered_raw_data[~filtered_raw_data['Post_ID'].isin(negative_authors_post_ids)]
    tab1, tab2 = st.tabs(["Submissions", "List of Removed Authors"])
    with tab2:
        st.markdown(f"### Removing <span style='color:#f70202'><b>{remove_negative_count}</b></span> Negative Authors ",unsafe_allow_html = True)
        st.dataframe(Array_remove_negative_authors,use_container_width= True)
        
    with tab1:
        st.markdown(f"### Submissions reduced from <span style='color:#f70202'><b>{filtered_raw_data.shape[0]}</b></span> to <span style='color:#f70202'><b>{filtered_raw_data_after_removal.shape[0]}</b></span>",unsafe_allow_html = True)
        generate_comparison_columns(filtered_raw_data, filtered_raw_data_after_removal, Type_data)
    

st.header(f"Raw dataset")
st.dataframe(filtered_raw_data, use_container_width=True)

st.header(f"Unique Authors")
st.dataframe(filtered_unique_authors_data, use_container_width=True)