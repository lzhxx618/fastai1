import streamlit as st
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader

@st.cache_data
def load_data():
    # Load the data
    data_df = pd.read_excel('data_test2.xlsx')


    # Load the foods
    foods_df = pd.read_excel('food.xlsx', header=None)
    foods_df.columns = ['food']
    foods_df.index.name = 'food_id'

    return data_df, foods_df

def train_model(data_df):
    # Create a Reader object
    reader = Reader(rating_scale=(0, 5))

    # Load the data into a Dataset object
    data = Dataset.load_from_df(data_df[['user_id', 'food_id', 'rating']], reader)

    # Build a full trainset from data
    trainset = data.build_full_trainset()

    # Train a SVD model
    algo = SVD()
    algo.fit(trainset)

    return algo

def recommend_foods(algo, data_df, foods_df, new_user_id, new_ratings):
    # Convert ratings from 0-5 scale to 5 to 10 scale
    new_ratings = {food_id: info['rating']*4 - 10 for food_id, info in new_ratings.items()}

    # Add new user's ratings to the data
    new_ratings_df = pd.DataFrame({
    'user_id': [new_user_id]*len(new_ratings),
    'food_id': list(new_ratings.keys()),
    'rating': list(new_ratings.values())
    })

    data_df = pd.concat([data_df, new_ratings_df])

    # Generate recommendations for the new user
    iids = data_df['food_id'].unique() # Get the list of all food ids
    iids_new_user = data_df.loc[data_df['user_id'] == new_user_id, 'food_id'] # Get the list of food ids rated by the new user
    iids_to_pred = np.setdiff1d(iids, iids_new_user) # Get the list of food ids the new user has not rated

    # Predict the ratings for all unrated foods
    testset_new_user = [[new_user_id, iid, 0.] for iid in iids_to_pred]
    predictions = algo.test(testset_new_user)

    # Get the top 5 foods with highest predicted ratings
    top_5_iids = [pred.iid for pred in sorted(predictions, key=lambda x: x.est, reverse=True)[:5]]
    top_5_foods = foods_df.loc[foods_df.index.isin(top_5_iids), 'food']

    return top_5_foods

def main():
    # Load data
    data_df, foods_df = load_data()

    # Choose an unused user_id for the new user
    new_user_id = data_df['user_id'].max() + 1

    # Randomly select 3 foods for the user to rate：进入页面能够随机显示3条笑话
    if 'initial_ratings' not in st.session_state:
        st.session_state.initial_ratings = {}
        random_foods = foods_df.sample(n=3)       #随机选取3条
        for food_id, food in zip(random_foods.index, random_foods['food']):
            st.session_state.initial_ratings[food_id] = {'food': food, 'rating': 3}

    # Ask user for ratings
    for food_id, info in st.session_state.initial_ratings.items():
        st.write(info['food'])
        info['rating'] = st.slider('Rate this food', 0, 5, step=1, value=info['rating'], key=f'init_{food_id}')    # 设置一个滑动条，用户能够拖动滑动条对这3条笑话进行评分

    # 设置一个按钮“Submit Ratings”，用户在点击按钮后，能够生成对该用户推荐的5条笑话
    if st.button('Submit Ratings'):
        # Add new user's ratings to the data
        new_ratings_df = pd.DataFrame({
            'user_id': [new_user_id] * len(st.session_state.initial_ratings),
            'food_id': list(st.session_state.initial_ratings.keys()),
            'rating': [info['rating'] for info in st.session_state.initial_ratings.values()]  # Convert scale from 0-5 to -10-10
        })
        data_df = pd.concat([data_df, new_ratings_df])
        # Train model
        algo = train_model(data_df)

        # Recommend foods based on user's ratings
        recommended_foods = recommend_foods(algo, data_df, foods_df, new_user_id, st.session_state.initial_ratings)

        # Save recommended foods to session state
        st.session_state.recommended_foods = {}
        for food_id, food in zip(recommended_foods.index, recommended_foods):
            st.session_state.recommended_foods[food_id] = {'food': food, 'rating': 3}

    # Display recommended foods and ask for user's ratings
    if 'recommended_foods' in st.session_state:
        st.write('We recommend the following foods based on your ratings:')
        # 显示基于用户评分所推荐的笑话
        for food_id, info in st.session_state.recommended_foods.items():
            st.write(info['food'])
            info['rating'] = st.slider('Rate this food', 0, 5, step=1, value=info['rating'], key=f'rec_{food_id}')

        #设置按钮“Submit Recommended Ratings”，点击按钮生成本次推荐的分数percentage_of_total，
        #计算公式为：percentage_of_total = (total_score / 25) * 100。。
        if st.button('Submit Recommended Ratings'):
            # Calculate the percentage of total possible score
            total_score = sum([info['rating'] for info in st.session_state.recommended_foods.values()])
            percentage_of_total = (total_score / 25) * 100
            st.write(f'Your percentage of total possible score: {percentage_of_total}%')

if __name__ == '__main__':
    main()