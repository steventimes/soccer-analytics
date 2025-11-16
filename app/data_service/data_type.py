from enum import Enum

class type_db_data(Enum):
    TEAM_PLAYER = 1
    COMPETITION_STANDING = 2
    SINGLE_TEAM = 3
    COMPETITION_MATCHES = 4
    TOP_SCORERS = 5
    MATCH_DETAILS = 6
    TEAM_MATCHES = 7
    
def competitions():
    competitions = ["CL","PL", "PPL", "DED", "BL1", "FL1", "SA", "PD", "BSA", "ELC", "EC", "WC"]
    competitions_id = [2001, 2021, 2017, 2003, 2002, 2015, 2019, 2014, 2013, 2016, 2018, 2000]
    competitions_mapping = dict(zip(competitions, competitions_id)) 
    return competitions_mapping