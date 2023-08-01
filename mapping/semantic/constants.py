class MapConstants:
    NON_SEM_CHANNELS = 6 + 34 # Number of non-semantic channels at the start of maps +  voxel height
    OBSTACLE_MAP = 0
    EXPLORED_MAP = 1 # 2d projection of all height channels
    CURRENT_LOCATION = 2 
    VISITED_MAP = 3 # only floor height
    BEEN_CLOSE_MAP = 4 # closely explored
    PROBABILITY_MAP = 5 # Probability of goal object being at location (i, j)
    VOXEL_START = 6 # Start of voxel height channels

# class ProbabilisticMapConstants:
#     NON_SEM_CHANNELS = 5  # Number of non-semantic channels at the start of maps
#     OBSTACLE_MAP = 0
#     EXPLORED_MAP = 1
#     CURRENT_LOCATION = 2
#     VISITED_MAP = 3
#     BEEN_CLOSE_MAP = 4
