#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Print multiple output per cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'


# In[2]:


## Importing libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()


# In[3]:


## Pandas Display options
pd.set_option('display.max_columns',0)
pd.set_option('display.max_colwidth',0)


# ## Read data

# In[4]:


## Reading relevant data
match_lvl_data = pd.read_csv('../campus_challenge_data/match_level_scorecard.csv')
batsman_lvl_data = pd.read_csv('../campus_challenge_data/batsman_level_scorecard.csv')
bowler_lvl_data = pd.read_csv('../campus_challenge_data/bowler_level_scorecard.csv')
train_data = pd.read_csv('../data/data_fin12apr/train_data.csv')
test_data = pd.read_csv('../data/data_fin12apr/test_data.csv')


# Printing shape and sample rows of each dataset

# In[5]:


match_lvl_data.shape
match_lvl_data.head(2)


# In[6]:


batsman_lvl_data.shape
batsman_lvl_data.head(2)


# In[7]:


bowler_lvl_data.shape
bowler_lvl_data.head(2)


# In[8]:


train_data.shape
train_data.head(2)


# In[9]:


test_data.shape
test_data.head(2)


# In[10]:


## Creating a binary winner column - 0 if team1 wins, else 1
train_data['winner_01'] = train_data.apply(lambda x: 0 if (x['team1']==x['winner']) else 1, axis=1)


# ## Code to plot RnP

# In[11]:


from matplotlib import pyplot as plt
plt.style.use('seaborn');
import re

def createRnP(X_12, feature, N=5, ylim_lb=0.3, ylim_ub=0.7):
    '''
    Rank and Plot of input feature on the input data. The y-axis shows %team1 wins in each bucket.
    
    Parameters-
    1. X_12: dataset to build the RnP on. 
    2. feature: Feature to build RnP of.
    3. N: number of bins on x-axis. Default 5.
    4. ylim_lb: lower bound of y axis on plot.
    5. ylim_ub: upper bound of y axis on plot.
    
    Output-
    1. Rank and Plot
    
    Returns- None
    '''
    df = X_12.copy()
    df[f'{feature}_bin'] = df[feature].rank(pct=True)//(1/N) # divide feature values for all games in 5 equi-volume buckets.
    df['count'] = 1
    df['team1_win%'] = df['winner_01'].apply(lambda x: 1-x) # invert winner_01 to get team1 winner indicator
    df['team2_win%'] = df['winner_01'].copy()
    df[f'{feature}_min'] = df[feature].copy()
    df[f'{feature}_max'] = df[feature].copy()
    df_g = df.groupby(f'{feature}_bin').agg({'team1_win%':'mean', 'team2_win%':'mean', 'count':'sum', f'{feature}_min':'min',\
                                            f'{feature}_max':'max'}).reset_index()
    N = min(N,df_g.shape[0])
    blue_bar = df_g['team1_win%'].values.tolist()
    ind = np.arange(N)
    # plotting starts
    plt.figure(figsize=(10,5));
    plt.bar(ind, blue_bar, label='Team 1 win%');
    plt.axhline(y=0.5, linewidth=0.5, color='k', linestyle = '--')
    xlabel = re.sub('team_','ratio_',feature)
    plt.xlabel(f'{xlabel} (team1 / team2) bins');
    plt.ylabel('Win %');
    plt.title(f'RnP - {feature} vs win');
    df_g['xticks'] = df_g.apply(lambda x: str(round(x[f'{feature}_min'],2)) + ' - ' + str(round(x[f'{feature}_max'],2)), axis=1)
    plt.xticks(ind, df_g['xticks']);
    plt.ylim([ylim_lb,ylim_ub]);
    plt.legend(loc='best');
    x2,x1 = blue_bar[-1],blue_bar[0]
    slope = x2/x1
    if slope < 1:
        slope = 1/slope
        x1,x2 = x2,x1
    print('slope:', round(x2,2),'/',round(x1,2), '= ',round(slope,2))
    plt.show();


# #### Helper function

# In[12]:


def giveLastNgamesPlayer(player_id, date, n, bat_or_bowl):
    '''
    Function to get last n games stats of a player before an input date.
    
    Input-
    1. player_id: id of the player to get historical data.
    2. date: date to look-back and get n games. Stats returned are before this input date.
    3. n: Number of historical games stats to return.
    4. bat_or_bowl: Kind of stats to return. {'bat': batting stats to return, 'bowl': bowling stats to return}
    
    Output-None
    
    Returns- dataframe having bowling/batting stats from last n games of a player before an input date. 
    The results are sorted by date.
    '''
    if bat_or_bowl == 'bat':
        df_topick = batsman_lvl_data
        id_col = 'batsman_id'
    else:
        df_topick = bowler_lvl_data
        id_col = 'bowler_id'
        
    return df_topick[(df_topick['match_dt']<date)&(df_topick[id_col]==float(player_id))]\
                .sort_values(by='match_dt', ascending=False).head(n)


# ## Feature creation

# ### 1. team_count_50runs_last15 <br>
# Ratio of number of 50s by players in team1 to number of 50s by players in team2 in last 15 games

# In[13]:


def no50sLastn(player_list, date, n):
    '''
    Function to get total number of 50s scored by players in the roster of a team in last n games.
    
    Input-
    1. player_list: ':' separated list of player ids in the roster of a team.
    2. date: match date of the game to calculate this feature.
    3. n: Number of games to look-back and create this feature.
    
    Output-None
    
    Returns- int value denoting sum of 50s scored by all players in the roster.
    '''
    
    player_list = str(player_list).split(':') # split string of ':' separated ids into a list of ids
    res_list = []
    for player in player_list: # loop over each player_id in roster
        df_rel = giveLastNgamesPlayer(player_id=player, date=date, n=n, bat_or_bowl='bat') # getting batting stats from last n games for each player.
        df_rel['gte_50runs'] = np.where(df_rel['runs']>=50, 1, 0) # binary indicator to denote whether the player scored a 50 in the game (runs>=50).
        res_list.append(np.nansum(df_rel['gte_50runs']))# Sum up number of 50s for the player and append to a list. We will do this for all players.
    return np.nansum(res_list)# Sum up values of the list which is sum of 50s by all players in the roster.


# In[14]:


# Computing number of 50 runs in last 15 games for team1 for train dataset.
train_data['team1_count_50runs_last15'] = train_data.progress_apply(lambda x: \
            no50sLastn(player_list=x['team1_roster_ids'], date=x['match_dt'], n=15), axis=1)
# Computing number of 50 runs in last 15 games for team2 for train dataset.
train_data['team2_count_50runs_last15'] = train_data.progress_apply(lambda x: \
            no50sLastn(player_list=x['team2_roster_ids'], date=x['match_dt'], n=15), axis=1)


# In[15]:


# Taking ratio of (number of 50 runs in last 15 games for team1) to (number of 50 runs in last 15 games for team2). Adding 1 to handle divide by zero exceptions.
train_data['team_count_50runs_last15'] = (train_data['team1_count_50runs_last15']+1)/(train_data['team2_count_50runs_last15']+1)
train_data.drop(columns=['team1_count_50runs_last15','team2_count_50runs_last15'], inplace=True) # dropping intermediate columns


# In[16]:


train_data.shape
train_data.tail(2)


# In[17]:


# RnP of team_count_50runs_last15 computed over the train data. Slope denotes ratio of right most bin to left most bin.
createRnP(train_data, 'team_count_50runs_last15')


# In[19]:


## Doing similar process for test dataset

test_data['team1_count_50runs_last15'] = test_data.progress_apply(lambda x: \
            no50sLastn(player_list=x['team1_roster_ids'], date=x['match_dt'], n=15), axis=1)
test_data['team2_count_50runs_last15'] = test_data.progress_apply(lambda x: \
            no50sLastn(player_list=x['team2_roster_ids'], date=x['match_dt'], n=15), axis=1)
test_data['team_count_50runs_last15'] = (test_data['team1_count_50runs_last15'])/(test_data['team2_count_50runs_last15']+1)
test_data.drop(columns=['team1_count_50runs_last15','team2_count_50runs_last15'], inplace=True)
test_data.shape
test_data.head(2)


# ### 2. team_winp_last5 <br>
# 
# Ratio of team1's win % to team2's win % in last 5 games

# In[20]:


def winpLastn(team_id, date, n):
    '''
    Get a team's win % in last n games. If a team has won 3 game out of their last 5, win% is 60%.
    
    Input-
    1. team_id: ID of the team to get their last n games and winner information from those games.
    2. date: match date from which to get last n historical games.
    3. n: look-back window of games.
    
    Output- None
    
    Returns- Float value denoting win% of the team in last n games.
    '''
    # filter out games with either team1/2_id as input team id, match_dt being before current game's date, sort desc by date, and get top n rows (games)
    df_rel = match_lvl_data[(match_lvl_data['match_dt']<date)&\
                      ((match_lvl_data['team1_id']==team_id)|(match_lvl_data['team2_id']==team_id))]\
                        .sort_values(by='match_dt', ascending=False).head(n) 
    win_count = df_rel[df_rel['winner_id']==team_id].shape[0] # count number of rows having winner as the input team
    if win_count == 0:
        return 0
    return round(win_count*100/df_rel.shape[0], 2) # return win% rounded to two decimal points


# In[21]:


# Compute team1's win% in last 5 games
train_data['team1_winp_last5'] = train_data.progress_apply(lambda x: \
                                  winpLastn(x['team1_id'], x['match_dt'], 5), axis=1)
# Compute team2's win% in last 5 games
train_data['team2_winp_last5'] = train_data.progress_apply(lambda x: \
                                  winpLastn(x['team2_id'], x['match_dt'], 5), axis=1)


# In[22]:


# Take the ratio of (team1's win% in their last 5 games)/(team2's win% in their last 5 games). Adding 1 to avoid divide by zero error
train_data['team_winp_last5'] = (train_data['team1_winp_last5']+1)/(train_data['team2_winp_last5']+1)
train_data.drop(columns=['team1_winp_last5', 'team2_winp_last5'], inplace=True) # drop intermediate columns


# In[23]:


train_data.shape
train_data.head(2)


# In[24]:


## Similar process for test data

test_data['team1_winp_last5'] = test_data.progress_apply(lambda x: \
            winpLastn(team_id=x['team1_id'], date=x['match_dt'], n=5), axis=1)
test_data['team2_winp_last5'] = test_data.progress_apply(lambda x: \
            winpLastn(team_id=x['team2_id'], date=x['match_dt'], n=5), axis=1)
test_data['team_winp_last5'] = (test_data['team1_winp_last5']+1)/(test_data['team2_winp_last5']+1)
test_data.drop(columns=['team1_winp_last5','team2_winp_last5'], inplace=True)
test_data.shape
test_data.head(2)


# In[25]:


# RnP of team_winp_last5 computed over the train data.
createRnP(train_data, 'team_winp_last5', ylim_ub=0.65)


# In[26]:


train_data.drop(columns=['winner_01'], inplace=True) # Dropping intermediate column made to plot RnP.


# ### 3. teamonly_avg_runs_last15 <br>
# team1's avg inning runs in last 15 games

# In[27]:


## derived feature computed using toss winner & toss decision to denote the inning team1 bats.
# If team1 won the toss and chose to bat or team2 won the toss and chose to bowl, the feature takes the value 1, else 2.
match_lvl_data['team1_bat_inning'] = np.where( ((match_lvl_data['team1']==match_lvl_data['toss winner'])&(match_lvl_data['toss decision']=='bat'))|\
                                               ((match_lvl_data['team2']==match_lvl_data['toss winner'])&(match_lvl_data['toss decision']=='field')) , 1, 2)


# In[28]:


match_lvl_data.head(2)


# In[29]:


def teamAvgRunsLastn(team_id, date, n):
    '''
    Function to calculate a team's average runs in their last n games.
    
    Input-
    1. team_id: ID of the team to calculate average runs.
    2. date: match date of the current game for which the feature is calculated.
    3. n: look-back window of games for the team.
    
    Output- None
    
    Return- Float value denoting average of runs scored by team1 in their last n games.
    '''
    # filter out games with either team1/2_id as input team_id, match date less than current game's input date, sort desc by date, and top n rows (games) returned
    df_rel = match_lvl_data[(match_lvl_data['match_dt']<date)&\
                      ((match_lvl_data['team1_id']==team_id)|(match_lvl_data['team2_id']==team_id))]\
                        .sort_values(by='match_dt', ascending=False).head(n)
    # combine two dataframes - one where input team is batting first, and another one where input team is batting second.
    df_rel = pd.concat([ df_rel[df_rel['team1_bat_inning']==1][['inning1_runs']].rename(columns={'inning1_runs':'runs'}), \
                         df_rel[df_rel['team1_bat_inning']==2][['inning2_runs']].rename(columns={'inning2_runs':'runs'}) ] )
    return df_rel['runs'].mean() # return mean of the combined dataframe.


# In[30]:


# Compute average runs scored by team1 in their last 15 games for train data.
train_data['team1only_avg_runs_last15'] = train_data.progress_apply(lambda x: \
                                  teamAvgRunsLastn(x['team1_id'], x['match_dt'], 15), axis=1)


# In[31]:


# Similarly for test data.
test_data['team1only_avg_runs_last15'] = test_data.progress_apply(lambda x: \
            teamAvgRunsLastn(x['team1_id'], x['match_dt'], 15), axis=1)
test_data.shape
test_data.head(2)


# In[32]:


train_data.shape
train_data.head(2)


# ### 4. teamone_winp_teamtwo_last15 <br>
# Team1's win percentage againts Team2 in last 15 games

# In[33]:


def winpCrossLastn(team1_id, team2_id, date, n):
    '''
    Function to compute team1's win% against team2 from the current game in their past n encounters.
    
    Input-
    1. team1_id: ID of team1 to calculate win% of.
    2. team2_id: ID of team2 to calculate win% against.
    3: date: match date of the current game for which the feature is to be calculated.
    4. n: look-back window of games for both these teams.
    
    Output- None
    
    Returns- Float value denoting team1's win% against team2 in their past n games against each other.
    '''
    # filter out games where either team1_id is input team1 and team2_id is input team2, or where team2_id is input team1 and team1_id is input team2.
    # Also, match date is less than current games's input date, sort desc by date and get top n rows (games)
    df_rel = match_lvl_data[(match_lvl_data['match_dt']<date)&\
                      (((match_lvl_data['team1_id']==team1_id)&(match_lvl_data['team2_id']==team2_id))|((match_lvl_data['team1_id']==team2_id)&(match_lvl_data['team2_id']==team1_id)))]\
                        .sort_values(by='match_dt', ascending=False).head(n)
    win_count = df_rel[df_rel['winner_id']==team1_id].shape[0] # Counting number of rows (games) where winner is input team1.
    if win_count == 0:
        return 0
    return round(win_count*100/df_rel.shape[0], 2) # return Float denoting team1's win% against team2 in past n games rounded to 2 decimal places.


# In[34]:


# Compute team1 win% against team2 in their past 15 encounters for train data.
train_data['team1_winp_team2_last15'] = train_data.progress_apply(lambda x: \
                                  winpCrossLastn(x['team1_id'], x['team2_id'], x['match_dt'], 5), axis=1)


# In[35]:


train_data.shape
train_data.head(2)


# In[36]:


# Similarly for test data.
test_data['team1_winp_team2_last15'] = test_data.progress_apply(lambda x: \
                                  winpCrossLastn(x['team1_id'], x['team2_id'], x['match_dt'], 5), axis=1)


# In[37]:


test_data.shape
test_data.head(2)


# ### 5. ground_avg_runs_last15 <br>
# average runs scored in the ground in last 15 games

# In[38]:


def avgRunsGround(ground_id, date, n):
    '''
    Function to calculate average runs scored in ground/venue.
    
    Input-
    1. ground_id: ID of the ground to calculate the feature for.
    2. date: match date of the current game to calculate the feature for.
    3. n: look-back window of games for the ground.
    
    Output- None
    
    Returns- Average runs scored in the ground.
    '''
    # filter out games with ground_id being the input ground_id and date earlier than current game's input date. Sort desc by date, and select top n rows (games).
    df_rel = match_lvl_data[(match_lvl_data['match_dt']<date)&(match_lvl_data['ground_id']==ground_id)].sort_values(by='match_dt', ascending=False).head(n)
    df_rel['avg_runs_inn'] = (df_rel['inning1_runs']+df_rel['inning2_runs'])/2 # take the mean of inning1_runs and inning2_runs in a separate column.
    return df_rel['avg_runs_inn'].mean() # Return the mean value of the computed column above.


# In[39]:


## Calculate average runs in the ground for last 15 games hosted in that venue for train data.
train_data['ground_avg_runs_last15'] = train_data.progress_apply(lambda x: \
                                  avgRunsGround(x['ground_id'], x['match_dt'], 15), axis=1)
## Similarly for test data.
test_data['ground_avg_runs_last15'] = test_data.progress_apply(lambda x: \
                                  avgRunsGround(x['ground_id'], x['match_dt'], 15), axis=1)


# In[40]:


train_data.shape
train_data.head(2)


# In[41]:


test_data.shape
test_data.head(2)


# ## Save

# In[57]:


# train_data.to_csv('train_data_with_samplefeatures', index=False)
# test_data.to_csv('test_data_with_samplefeatures', index=False)



import pandas as pd
import sys


# Instructions for participants :
'''
Participants can use this code to run on labeled train/out-of-sample data to mimic evaluation process.
### Datasets required:
This script takes in 3 files as follows:

primary_submission.csv -> This contains the match_id, dataset_type, win_pred_team_id, win_pred_score, train_algorithm, is_ensemble, train_hps_trees, train_hps_depth, train_hps_lr, *top 10 feature values. This is file submitted by participant.
secondary_submission.csv -> This contains feature_name, feature_description, model_feature_importance_rank, model_feature_importance_percentage, feature_correlation_dep_var. This is file submitted by participant.
dep_var.csv    -> This contains match_id, dataset_type, win_team_id. Participants can generate from the labeled train data.

Please ensure that the predicted_score column does not have any null columns and the column names are exactly matching as above.
Please ensure that all these files are stored as ',' separated csv files.

### How to use:
To use this, first open the command line terminal, and call evaluation code script by passing the locations of submission and actual files respectively.
Sample example of using commandline for running the script: 

python Evaluation_Code.py path_to_primary_submission_file path_to_secondary_submission_file path_to_DepVar_file
'''


def checkDataType1(df):
    assert (df['match id'].isna().sum() == 0), 'match id should not have NaNs'
    assert (df['match id'].dtype == 'int64'), ('match id is not int64 type')
    assert df['win_pred_team_id'].isna().sum(
    ) == 0, 'win_pred_team_id should not have NaNs'
    assert df['win_pred_team_id'].dtype == 'int64', (
        'win_pred_team_id is not int64 type')
    assert df['win_pred_score'].isna().sum(
    ) == 0, 'win_pred_score should not have NaNs'
    assert df['win_pred_score'].dtype == 'float64', (
        'win_pred_score is not float64 type')
    assert df['train_algorithm'].isna().sum(
    ) == 0, 'train_algorithm should not have NaNs'
    assert df['train_algorithm'].dtype == 'object', (
        'train_algorithm is not object type')
    assert df['is_ensemble'].isna().sum(
    ) == 0, 'is_ensemble should not have NaNs'
    assert df['is_ensemble'].dtype == 'object', (
        'is_ensemble is not object type')
    assert df['train_hps_trees'].isna().sum(
    ) == 0, 'train_hps_trees should not have NaNs'
    assert df['train_hps_depth'].isna().sum(
    ) == 0, 'train_hps_depth should not have NaNs'
    assert df['train_hps_lr'].isna().sum(
    ) == 0, 'train_hps_lr should not have NaNs'
    return None


def checkDataType2(df):
    assert df['feat_id'].isna().sum() == 0, 'feat_id should not have NaNs'
    assert df['feat_id'].dtype == 'int64', ('feat_id is not int type')
    assert df['feat_name'].isna().sum() == 0, 'feat_name should not have NaNs'
    assert df['feat_name'].dtype == 'object', ('feat_name is not object type')
    assert df['feat_description'].isna().sum(
    ) == 0, 'feat_description should not have NaNs'
    assert df['feat_description'].dtype == 'object', (
        'feat_description is not object type')
    assert df['model_feat_imp_train'].isna().sum(
    ) == 0, ' model_feat_imp_train should not have NaNs'
    assert df['model_feat_imp_train'].dtype == 'float64', (
        'model_feat_imp_train is not float type')
    assert df['feat_rank_train'].isna().sum(
    ) == 0, 'feat_rank_train should not have NaNs'
    assert df['feat_rank_train'].dtype == 'int64', (
        'feat_rank_train is not int64 type')
    return None


def getAccuracy(df):
    return round(df[df['winner_id'] == df['win_pred_team_id']].shape[0]*100/df.shape[0], 4)

if len(sys.argv) != 4:
  sys.exit("Please pass three files only as mentioned in the Instructions.")

# Location of submission file. Header here should include match_id, dataset_type, win_team_id. The file should be comma separated.
input1_address = sys.argv[1]
df_input1 = pd.read_csv(input1_address, sep=",", header=0)

input2_address = sys.argv[2]
df_input2 = pd.read_csv(input2_address, sep=",", header=0)

# For participants Team : Location of Dependent Variable file. Header here would be match_id, dataset_type, win_team_id. Participants can generate from the labeled train data. These files are comma separated
round_eval = sys.argv[3]
df_round = pd.read_csv(round_eval, sep=",", header=0)

assert set(['match id', 'dataset_type', 'win_pred_team_id', 'win_pred_score', 'train_algorithm', 'is_ensemble', 'train_hps_trees',
           'train_hps_depth', 'train_hps_lr']).issubset(set(df_input1.columns.tolist())), 'Required columns not present in primary submission file'
assert set(['indep_feat_id1', 'indep_feat_id2', 'indep_feat_id3', 'indep_feat_id4', 'indep_feat_id5', 'indep_feat_id6', 'indep_feat_id7', 'indep_feat_id8',
           'indep_feat_id9', 'indep_feat_id10']).issubset(set(df_input1.columns.tolist())), 'Required indepedent feature columns not present in primary submission file'
assert set(['feat_id', 'feat_name', 'feat_description', 'model_feat_imp_train', 'feat_rank_train']).issubset(
    set(df_input2.columns.tolist())), 'Required columns not present in secondary submission file'

checkDataType1(df_input1)
checkDataType2(df_input2)

assert df_input1.shape[0] == df_input1.drop_duplicates(
    'match id').shape[0], 'Input file should be unique on match id'
# assert df_input1.shape[
#     0] == 1219, f'Input file size number of rows incorrect. Expected rowsize 1219 not equal to uploaded data rowsize {df_input1.shape[0]}'
assert df_input1.shape[1] == 19, 'Input file number of columns not correct. '
assert (df_input1.win_pred_score.min() >= 0) & (df_input1.win_pred_score.max(
) <= 1), 'Win prediction score should be in range [0,1]'
assert df_input1['train_algorithm'].nunique(
) == 1, 'only one algorithm can be used for all data'
assert (len(df_input1['is_ensemble'].unique().tolist()) == 1) & ((df_input1['is_ensemble'].unique().tolist()[
    0] == 'yes') | (df_input1['is_ensemble'].unique().tolist()[0] == 'no')), 'is_ensemble can take only \'yes\' or \'no\''
assert df_input1.apply(lambda x: 0 if (len(str(x['train_algorithm']).split(';')) == len(str(x['train_hps_trees']).split(';'))) &
                       (len(str(x['train_algorithm']).split(';')) == len(str(x['train_hps_depth']).split(';'))) & (len(str(x['train_algorithm']).split(';')) == len(str(x['train_hps_lr']).split(';'))) else 1, axis=1).max() == 0, 'number of fields in algorithm & hyper-parameters column should be same.'

'''
shape_before_join = df_round.shape[0]

r1_size = df_input1[df_input1['dataset_type'] == 'r1'].shape[0]
assert (r1_size ==
        df_round.shape[0]), f'R1 data size in input file is incorrect. Expected rowsize 271 not equal to r1 dataset_type present {r1_size}'
'''

# merging predicted file and dependent variable file
eval_data = pd.merge(df_round, df_input1, on=[
                     'match id'], how='inner').drop_duplicates()
assert (eval_data.shape[0] == df_round.shape[0]
        ), 'match ids in submission template does not match eval data'

print('All checks passed...')
print('Accuracy: ', round(getAccuracy(eval_data), 2))
