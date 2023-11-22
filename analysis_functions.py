from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Read the datasets: 
#agents = pd.read_csv('data/export_per_agent.csv')
#areas = pd.read_csv('data/export_per_area.csv')
#cross = pd.read_csv('data/export_per_area_agent.csv')
# Read the geography dataset, that contains all level-4 areas for root_id=1 and 3:
#geography_df = pd.read_csv('data/geography.csv')


def prepare_datasets(membership_slots):
    '''
    Prepare all the datasets that we will need. 
    '''

    global agents
    global areas 
    global cross
    global geography_df

    brand_slots, brandPlus_slots, expert_slots = membership_slots
    
    # Keep only columns we need:
    agents_df = agents.loc[:, ['agent_id', 'membership_current']].copy().rename(columns={'membership_current': 'membership'})
    areas_df = areas.loc[:, ['lvl4_id', 'lvl4_name', 'agents', 'entrustments']].copy()
    cross_df = cross.loc[:, ['lvl4_id', 'agent_id', 'properties']].copy()
    
    conditions = [(agents_df['membership']=='Brand'), 
                  (agents_df['membership']=='Brand Plus'), 
                  (agents_df['membership']=='Expert')]
    slot_values = [brand_slots, brandPlus_slots, expert_slots]
    agents_df['membership_slots'] = np.select(conditions, slot_values, default=0)
    
    # Add membership_slots on cross_df:
    cross_df = cross_df.merge(agents_df.loc[:, ['agent_id', 'membership_slots']], on='agent_id')
    
    return(agents_df, areas_df, cross_df, geography_df)


def get_lvl4_thresholds(df, n_specialists=2):
    '''
    For each level-4 area find the threshold such that there will be n_specialists
    in each area, and adds the column 'lvl4_threshold'.
    
    Method: For each area we sort the properties of the agents in ascending order, and 
            take the 2nd value (or in general n_specialists value) from the end to find
            the threshold.    
    Notes: 
         1. We have considered that the threshold is 'greater than or equal'
         2. If an area has less than 2 (or less than 'n_specialists' in general) agents then 
            we consider as property threshold the first number of the sorted list, i.e. the 
            minimum number.
    '''
    
    thresholds_dict = dict()
    for area_id in list(df.lvl4_id.unique()):
        agents_properties = list(df[df['lvl4_id']==area_id].properties.sort_values(ascending=True))
        
        if len(agents_properties)>=n_specialists:
            # If threshold is 'greater than or equal', in order to be a specialist:
            threshold = agents_properties[-n_specialists]
            #threshold = agents_properties[-n_specialists-1] # for 'greater than'
        else:
            # In case an area has less than 2 (or less than 'n_specialists' in general) agents, 
            # use the number of properties of the first agent (low to high) :
            threshold = agents_properties[0]
            
        thresholds_dict[area_id] = threshold
        
    # Merge with 'geography' table to also get the missing areas. Put all in a dataframe:
    df_threshold = (pd.DataFrame(list(thresholds_dict.items()), columns=['lvl4_id', 'lvl4_threshold'])
                     .merge(geography_df.lvl4_id, how='right', on='lvl4_id')
                     .fillna(value=0)
                     .astype({'lvl4_threshold': int})
                     .sort_values('lvl4_threshold', ascending=False)
                     .reset_index(drop=True) )
    df = df.merge(df_threshold, on='lvl4_id')
    
    return(df)


def assign_areas_to_tiers(df, tier_thresholds): 
    '''
    Given the predefined thresholds, create Tiers and assign each area to a Tier.
    '''
    
    t1, t2, t3, t4 = tier_thresholds
    
    condition_1 = (df['lvl4_threshold'] >= t1)
    condition_2 = (df['lvl4_threshold'] >= t2) & (df['lvl4_threshold'] < t1)
    condition_3 = (df['lvl4_threshold'] >= t3) & (df['lvl4_threshold'] < t2)
    condition_4 = (df['lvl4_threshold'] >= t4) & (df['lvl4_threshold'] < t3)

    conditions = [condition_1, condition_2, condition_3, condition_4]
    tiers = list(range(1, 4+1))
    
    # Add a new column for the 'Tier':
    df['Tier'] = np.select(conditions, tiers)
    
    # Set the universal threshold for each Tier (which is the minimum border of each Tier): 
    conditions = [(df['Tier']==1), (df['Tier']==2), (df['Tier']==3), (df['Tier']==4)]
    tier_thresholds = [t1, t2, t3, t4]
    
    df['Tier threshold'] = np.select(conditions, tier_thresholds, default=0)
    
    return(df)


def display_Tiers_countplot(df):
    '''
    Creates a countplot displaying the number of level-4 areas on each Tier.
    '''
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(4, 3))

    plot = sns.countplot(data=df, x='Tier')

    # Annotate each bar with its count value
    for p in plot.patches:
        plot.annotate(format(p.get_height(), '.0f'), 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha = 'center', va = 'center', 
                      xytext = (0, 6), 
                      textcoords = 'offset points')
    plt.title('Number of level-4 areas on \neach Tier')

    plt.show()
        
        
def display_Tiers_countplot_with_leads(areas_df, tiers_df, min_leads=40):
    '''
    Creates a countplot displaying the number of level-4 areas on each Tier, and the number of areas 
    that had more than 'min_leads', for each Tier. 
    '''

    ddf = (areas_df.merge(tiers_df.loc[:, ['lvl4_id', 'Tier']], on='lvl4_id', how='right')
                   .assign(high_leads=lambda x: np.where(x['entrustments']>=min_leads, x['lvl4_id'], np.nan)) )
    df1 = pd.DataFrame(ddf.groupby(by='Tier').high_leads.nunique()).reset_index()
    df2 = pd.DataFrame(ddf.groupby(by='Tier').lvl4_id.nunique()).reset_index()
    combined_df = df1.merge(df2, on='Tier')

    fig, ax = plt.subplots(figsize=(6, 3))
    bar_positions = np.arange(len(combined_df['Tier']))
    bars_set1 = ax.bar(bar_positions, combined_df['lvl4_id'], label='Total areas', alpha=0.7)
    # Plotting the second set of bars on top of the first
    bars_set2 = ax.bar(bar_positions, combined_df['high_leads'], label=f'Areas with leads>={min_leads}', 
                                                           alpha=0.7, bottom=combined_df['lvl4_id'])

    # Add total numbers to each bar
    for bars, values in zip([bars_set1, bars_set2], [combined_df['lvl4_id'], combined_df['high_leads']]):
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                    f'{value}', ha='center', va='center', color='black')


    plt.xlabel('Tier')
    plt.ylabel('level-4 areas')
    plt.title('Number of areas in each Tier')

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(combined_df['Tier'])

    plt.legend()
    plt.show()
    
        
def get_tiers_df(df):
    '''
    Returns a dataframe with the level-4 area, its threshold, the Tier it belongs, and the Tier's threshold.
    ''' 
    
    tiers_df = (df.loc[:, ['lvl4_id', 'lvl4_threshold', 'Tier', 'Tier threshold']]
                  .drop_duplicates()
                  .reset_index(drop=True) )
    
    return(tiers_df)


def get_eligible_info(df):
    '''
    Given the cross dataset as input, it creates a new column holding the information of whether an 
    agent is eligible on the specific level-4 area. 
    '''
    
    df = df.assign(eligible=lambda x: np.where(x['properties'] >= x['Tier threshold'], 1, 0))
    
    
    total_agents = df.agent_id.nunique()
    total_eligible = df.query('eligible == 1').agent_id.nunique()
    perc_eligible = round(100*total_eligible/total_agents, 1)
    perc_not_eligible = 100-perc_eligible

    print(f"Total agents eligible in at least one level-4 area: \t{total_eligible} ({perc_eligible}%)")
    print(f"Total agents that are not eligible in any area: \t{total_agents-total_eligible} ({perc_not_eligible}%)")
    
    return(df)


def get_areas_is_eligible_df(df):
    '''
    Returns a dataframe with the total number of level-4 areas that each agent was found to be eligible.

    '''
    
    eligible_df = df.copy()
    areas_is_eligible_df = (pd.DataFrame(eligible_df.groupby('agent_id')
                              .eligible.sum()
                              .sort_values(ascending=False))
                              .rename(columns={'eligible':'areas_being_eligible'})
                              .reset_index() )
    
    return(areas_is_eligible_df)


def get_eligibles_per_area_df(df):
    '''
    Returns a dataframe with the number of eligible agents on each level-4 area. 
    '''
    
    eligibles_per_area_df = (pd.DataFrame(df.groupby(by='lvl4_id').eligible.sum())
                                            .reset_index()
                                            .merge(geography_df.loc[:, ['lvl4_id']], on='lvl4_id')
                                            .sort_values(by='eligible', ascending=False)
                                            .reset_index(drop=True)
                                            .loc[:, ['lvl4_id', 'eligible']])

    return(eligibles_per_area_df)



def get_specialists_per_area_df(df):
    '''
    Returns a dataframe with the number of eligible agents and specialists on each level-4 area.
        * As an extra information the number of agents previously active on the area, as well 
          as the number of leads (entrustments) is given, 
        * For finding the specialists on each area, we take into account the membership slots of
          each agent (given from his membership package) and make the assumption that he will 
          choose the areas that he has the most properties.
    
    It includes all the level-4 geographies, including those that we had no data on the cross dataset.
    '''
     
    global areas
    areas_df = areas.loc[:, ['lvl4_id', 'lvl4_name', 'agents', 'entrustments']].copy()

    # Take the (agents,areas) couples for which an agent is eligible in this area:
    specialists_df = df.query('eligible == 1').copy()

    all_area_preferences = list()
    for agent_id in list(specialists_df.agent_id.unique()):
        slots = specialists_df[specialists_df['agent_id']==agent_id].iloc[0].membership_slots
        # We assume that his slots will be filled with the areas he has the most properties in:
        areas_preference = list(specialists_df[specialists_df['agent_id']==agent_id]\
                                                .sort_values(by='properties', ascending=False)\
                                                .head(slots)\
                                                .lvl4_id)
        ### If there are ties: 
        ### Use 'ordered_lvl4_by_properties' to treat them:
        #ordered_lvl4_by_properties = areas[['lvl4_id', 'properties']].sort_values(by='properties', ascending=False)\
        #                                                         .reset_index(drop=True)
        all_area_preferences.extend(areas_preference)

    # Count frequency of each area, and put in dataframe: 
    area_freq = Counter(all_area_preferences)
    specialists_per_area_df = (pd.DataFrame(list(dict(area_freq).items()), columns=['lvl4_id', 'specialists'])
                                 .reset_index(drop=True)
                                 # merge with 'geography' to get the missing level-4 areas:
                                 .merge(geography_df, how='right', on='lvl4_id')
                                 .merge(areas_df[['lvl4_id', 'agents', 'entrustments']], how='left', on='lvl4_id')
                                 .fillna(value=0)
                                 .astype({'specialists': int})
                                 .rename(columns={'agents': 'agents_previously'})
                                 .loc[:, ['lvl4_id', 'lvl4_name', 'specialists', 'agents_previously', 'entrustments']]
                                 .sort_values(by='specialists', ascending=False)
                                 .reset_index(drop=True) )
    
    # Add the column of the eligible agents:
    eligibles_per_area_df = get_eligibles_per_area_df(df)
    specialists_per_area_df = specialists_per_area_df.merge(eligibles_per_area_df, on='lvl4_id')
    
    ordered_cols = ['lvl4_id', 'lvl4_name', 'eligible', 'specialists', 'agents_previously', 'entrustments']
    specialists_per_area_df = specialists_per_area_df.loc[:, ordered_cols]
        
    return(specialists_per_area_df)


def get_areas_no_specialists_df(specialists_per_area_df):
    '''
    Returns a dataframe with the level-4 areas that have no specialists and the entrustments of each area.
    '''
    
    areas_no_specialists_df = (specialists_per_area_df
                                        .query('specialists == 0') 
                                        .sort_values(by='entrustments', ascending=False)
                                        .assign(entrustments_perc=lambda x: round(100 * x['entrustments'] /
                                                                    specialists_per_area_df['entrustments'].sum(), 1))
                                        .assign(entrustments_perc=lambda x: x['entrustments_perc'].astype(str) + ' %') 
                                        .reset_index(drop=True))

    unanswered = areas_no_specialists_df.entrustments.sum()
    perc = round(100*unanswered/specialists_per_area_df.entrustments.sum(), 1)
    no_spec_perc = round(100*areas_no_specialists_df.shape[0]/(specialists_per_area_df.lvl4_id.nunique()), 1)
    
    print(f"\nAreas with no specialists: \t\t\t\t{areas_no_specialists_df.shape[0]}  ({no_spec_perc}%)")
    print(f"\nTotal entrustments left unanswered:  \t\t\t{unanswered}  ({perc}%)")    
    
    return(areas_no_specialists_df)


def get_leads_no_spec_areas_df(areas_no_specialists_df):
    '''
    Returns a dataframe, with the count and percentage of areas with no specialists devided on whether
    they had leads (entrustments) or not. 
    '''
    
    leads_no_spec_areas_df = (pd.DataFrame(areas_no_specialists_df.copy()
                                       .assign(leads=lambda x: np.where(x['entrustments']>0, 1, 0))
                                       .value_counts('leads'))
                                       .rename({0:'areas_count'}, axis=1)
                                       .reset_index()
                                       .replace({0: 'no leads', 1: 'with leads'})
                                       .assign(areas_perc=lambda x: round(100*x['areas_count']/x['areas_count'].sum(), 1))
                                       .assign(areas_perc=lambda x: x['areas_perc'].astype(str) + ' %') 
                                       .set_index('leads') )

    return(leads_no_spec_areas_df)


def get_empty_slots_df(areas_is_eligible_df, cross_df):
    '''
    Returns a dataframe for each number of slots not used with the count of agents and percentage. 
    '''
        
    df = cross_df.copy()
    
    empty_slots_df = (areas_is_eligible_df
                         .copy()
                         .reset_index()
                         .merge(df.loc[:, ['agent_id', 'membership_slots']].drop_duplicates())
                         .assign(empty_granted_slots=lambda x: x['membership_slots'] - x['areas_being_eligible'])
                         .assign(empty_granted_slots=lambda x: x['empty_granted_slots'].mask(x['empty_granted_slots'] <= 0, 0))
                         .reset_index(drop=True)
                         .groupby(by='empty_granted_slots').count()
                         .loc[:, ['agent_id']]
                         .assign(perc=lambda x: round(100 * x['agent_id'] / df['agent_id'].nunique(), 1))
                         .rename(columns={'agent_id': 'agents_count', 'perc': 'agents_perc'})
                         .assign(agents_perc=lambda x: x['agents_perc'].astype(str) + ' %') )
    
    return(empty_slots_df)
