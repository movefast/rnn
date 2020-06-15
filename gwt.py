import tiles3 as tc
import numpy as np

# Tile Coding Function [Graded]
class GridWorldTileCoder:
    def __init__(self, iht_size=4096, num_tilings=8, num_tiles=8):
        """
        Initializes the MountainCar Tile Coder
        Initializers:
        iht_size -- int, the size of the index hash table, typically a power of 2
        num_tilings -- int, the number of tilings
        num_tiles -- int, the number of tiles. Here both the width and height of the
                     tile coder are the same
        Class Variables:
        self.iht -- tc.IHT, the index hash table that the tile coder will use
        self.num_tilings -- int, the number of tilings the tile coder will use
        self.num_tiles -- int, the number of tiles the tile coder will use
        """
        self.iht = tc.IHT(iht_size)
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
    
    def get_tiles(self, position, is_door):
        """
        Takes in a position and velocity from the mountaincar environment
        and returns a numpy array of active tiles.
        
        Arguments:
        returns:
        tiles - np.array, active tiles
        """
        # Set the max and min of position and velocity to scale the input
        # POSITION_MIN
        # POSITION_MAX
        # VELOCITY_MIN
        # VELOCITY_MAX
        ### BEGIN SOLUTION
        POSITION_MIN = 0
        POSITION_MAX = 49
        DOOR_MIN = 0
        DOOR_MAX = 1
        ### END SOLUTION
        
        # Scale position and velocity by multiplying the inputs of each by their scale
        
        ### BEGIN SOLUTION
        position_scale = self.num_tiles / (POSITION_MAX - POSITION_MIN)
        is_door_scale = self.num_tiles / (DOOR_MAX - DOOR_MIN)
        ### END SOLUTION
        
        # get the tiles using tc.tiles, with self.iht, self.num_tilings and [scaled position, scaled velocity]
        tiles = tc.tiles(self.iht, self.num_tilings, [position * position_scale, is_door * is_door_scale])
        
        return np.array(tiles)