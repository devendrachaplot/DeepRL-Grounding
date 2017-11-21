# Difference in size of the objects to be considered larger or smaller
SIZE_THRESHOLD = 100

# Maximum distance from the object to receive a reward
REWARD_THRESHOLD_DISTANCE = 40

# Rewards for reaching the correct and wrong objects
CORRECT_OBJECT_REWARD = 1.0
WRONG_OBJECT_REWARD = -0.2

# Size of the map
MAP_SIZE_X = 384
MAP_SIZE_Y = 384

# Map offsets in doom coordinates
Y_OFFSET = 320
X_OFFSET = 0

# Margin to avoid objects overlapping with the walls
MARGIN = 32

# Distance between y-coordinates of two objects in Easy and Medium environments
OBJECT_Y_DIST = 64

# X-coordinate of all objects in the Easy environment
EASY_ENV_OBJECT_X = 256

# Range of x coordinates of all objects in the Medium environment
MEDIUM_ENV_OBJECT_X_MIN = 192
MEDIUM_ENV_OBJECT_X_MAX = 352

# Minimum distance between any two objects or an object and
# the agent in the Hard environment
HARD_ENV_OBJ_DIST_THRESHOLD = 90
