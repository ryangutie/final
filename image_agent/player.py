import numpy as np
from .model import PuckDetector, load_model
import torchvision.transforms.functional as F
import math
from collections import deque


class HockeyPlayer:
    def __init__(self, player_id=0):
        # Initialize player with an ID and default values
        self.kart = 'xue'  # Default kart model
        self.state = 'kickoff'  # Initial game state
        self.player_id = player_id
        # History tracking for the kart's and puck's locations and actions
        self.past_kart_locs = deque(maxlen=5)
        self.past_puck_locs = deque(maxlen=5)
        self.past_state = deque(maxlen=5)
        self.past_actions = deque(maxlen=5)
        # Flags to manage game states
        self.state_lock = False
        self.state_lock_turns = 0
        self.stuck_count = 0
        self.search_count = 0
        # Current and target velocity of the kart
        self.current_vel = 0
        self.target_vel = 25
        self.last_known_puck = []  # Last known puck position
        self.model = load_model().eval()  # Load the trained model


        self.team = player_id % 2
        if self.team == 0:
            self.position = (player_id / 2) % 2
            # Goal positions for the teams
            self.our_goal_left = (-10, -64)
            self.our_goal_center = (0, -64)
            self.our_goal_right = (10, -64)
            self.their_goal_left = (-10, 64)
            self.their_goal_center = (0, 64)
            self.their_goal_right = (10, 64)
        else:
            self.position = (player_id - 1 / 2) % 2
            self.our_goal_left = (-10, 64)
            self.our_goal_center = (0, 64)
            self.our_goal_right = (10, 64)
            self.their_goal_left = (-10, -64)
            self.their_goal_center = (0, -64)
            self.their_goal_right = (10, -64)
        # Determine offensive or defensive role
        if player_id // 2 == 0:
            self.role = 'offense'
        else:
            self.role = 'defense'

    def act(self, image, player_info):
        # Prepare the default action settings
        action = {'acceleration': 1, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
        self.current_vel = np.linalg.norm(player_info.kart.velocity)  # Calculate the current velocity
        if len(self.past_actions) > 0:
            action = self.past_actions[-1]  # Use the last action if available

        # Transform the current image to tensor and predict puck location using the model
        image_transform = F.to_tensor(image)[None]
        self.image_puck_loc = (self.model(image_transform).detach().cpu().numpy())[0]
        self.puck_loc = self.image_puck_loc

        # Update kart and puck locations
        self.kart_loc = self.to_numpy(player_info.kart.location)
        self.kart_front = self.to_numpy(player_info.kart.front)

        # Check if the kart is stuck and needs to reset
        if len(self.past_kart_locs) != 0 and self.check_reset(self.kart_loc):
            self.state_lock = False

        # Determine and set the current state of the kart based on game dynamics
        if not self.state_lock:
            self.state = self.set_state(self.kart_loc, self.kart_front, self.puck_loc)
            if self.state == 'kickoff':
                action = self.kickoff_action(self.puck_loc)
            elif self.state == 'in_goal':
                action = self.getOutOfGoal(action)
            elif self.state == 'attack':
                action = self.attack_action(self.kart_loc, self.kart_front, self.puck_loc, action)
                self.last_known_puck = self.image_puck_loc
            elif self.state == 'searching':
                action = self.searching_action(self.kart_loc, self.kart_front, action)
            elif self.state == 'stuck':
                action = self.stuck_action(self.kart_loc, action)
        else:
            action = self.past_actions[-1]
            self.state_lock_turns -= 1
            if self.state_lock_turns == 0:
                self.state_lock = False

        # Update histories of locations, states, and actions
        self.past_kart_locs.append(self.kart_loc)
        self.past_puck_locs.append(self.puck_loc)
        self.past_state.append(self.state)
        self.past_actions.append(action)

        # Adjust acceleration based on the current velocity and state
        ratio = self.current_vel / self.target_vel
        if self.state != 'kickoff':
            if ratio <= 0.5:
                action['acceleration'] = 1
            elif ratio <= 0.7:
                action['acceleration'] = 0.7
            else:
                action['acceleration'] = 0
        return action

    # Function to determine the current state based on kart's position and puck's location
    def set_state(self, kart_loc, kart_front, puck_loc):
        if self.kickoff(kart_loc):
            self.kickoff_timer = 0
            return 'kickoff'
        self.kickoff_timer += 1
        if self.state == 'kickoff' and self.kickoff_timer < 33:
            return 'kickoff'
        if self.stuck(kart_loc):
            return 'stuck'
        elif self.inGoal(kart_loc):
            return 'in_goal'
        elif self.searching(puck_loc):
            return 'searching'
        return 'attack'

    # Helper functions to determine whether the kart is stuck, to calculate positions, etc.
    def kickoff(self, kart_loc):
        if len(self.past_kart_locs) == 0:
            return True
        return self.check_reset(kart_loc)

    def kickoff_action(self, puck_loc):
        action = {'acceleration': 1, 'steer': 4 * puck_loc[0], 'brake': False, 'nitro': True}
        x = self.kart_loc[0]
        y = self.kart_loc[-1]
        if x > 3:
            action['steer'] = -.32
        elif x < -3:
            action['steer'] = .32
        return action

    def inGoal(self, kart_loc):
        if abs(kart_loc[0]) < 10 and ((kart_loc[1] > 63.8) or (kart_loc[1] < -63.8)):
            self.state_lock = True
            self.state_lock_turns = 10
            return True
        return False

    def getOutOfGoal(self, action):
        if self.kart_loc[1] > 0:
            if self.kart_front[1] - self.kart_loc[1] > -.3:
                action['acceleration'] = 0
                action['brake'] = True
                if self.last_known_puck[0] < self.kart_loc[0]:
                    action['steer'] = 1
                else:
                    action['steer'] = -1
            else:
                action['acceleration'] = 1
                action['brake'] = False
                if self.last_known_puck[0] > self.kart_loc[0]:
                    action['steer'] = -1
                else:
                    action['steer'] = 1
        else:
            if self.kart_front[1] - self.kart_loc[1] < .3:
                action['acceleration'] = 0
                action['brake'] = True
                if self.last_known_puck[0] < self.kart_loc[0]:
                    action['steer'] = -1
                else:
                    action['steer'] = 1
            else:
                action['acceleration'] = 1
                action['brake'] = False
                if self.last_known_puck[0] < self.kart_loc[0]:
                    action['steer'] = 1
                else:
                    action['steer'] = -1
        if abs(self.kart_loc[1]) > 69:
            action['steer'] = action['steer'] * ((10 - abs(self.kart_loc[0])) / 10)
        action['nitro'] = False
        return action

    def stuck(self, kart_loc):
        no_move = (abs(self.kart_loc - self.past_kart_locs[-1]) < 0.02).all()
        no_vel = self.current_vel < 2.0
        no_try_move = (self.past_actions[0]['brake'] == False and self.past_actions[0]['acceleration'] != 0)
        danger_zone = abs(self.kart_loc[0]) >= 45 or abs(self.kart_loc[1]) >= 63.5
        if no_move and no_vel and no_try_move:
            if self.stuck_count < 5:
                self.stuck_count += 1
            else:
                self.stuck_count = 0
                self.state_lock = True
                if self.current_vel > 10.0 and self.past_actions[-1]['acceleration'] > 0:
                  self.state_lock_turns = 10
                else:
                    self.state_lock_turns = 7
                return True
        if no_move and no_vel and (danger_zone):
            self.state_lock = True
            if self.current_vel > 10.0 and self.past_actions[-1]['acceleration'] > 0:
              self.state_lock_turns = 10
            else:
                self.state_lock_turns = 7
            return True
        return False

    def stuck_action(self, kart_loc, action):
        if self.kart_loc[1] < 0:
            if self.kart_front[1] - self.kart_loc[1] < 0:
                action['acceleration'] = 0
                action['brake'] = True
            else:
                action['acceleration'] = 1
        else:
            if self.kart_front[1] - self.kart_loc[1] > -0.001:
                action['acceleration'] = 0
                action['brake'] = True
            else:
                action['acceleration'] = 1
        if abs(self.kart_loc[0]) >= 45:
            if action['acceleration'] > 0:
                action['steer'] = np.sign(self.kart_loc[0]) * -1
            else:
                action['steer'] = np.sign(self.kart_loc[0]) * 1
        else:
            if self.last_known_puck[1] > self.kart_loc[1]:
                if self.kart_loc[0] < 0:
                    action['steer'] = 1
                else:
                    action['steer'] = -1
            elif self.last_known_puck[1] < self.kart_loc[1]:
                if self.kart_loc[0] < 0:
                    action['steer'] = -1
                else:
                    action['steer'] = 1
        action['nitro'] = False
        return action

    def searching(self, puck_loc):
        threshold = -.8
        if puck_loc[1] < threshold and self.search_count < 3:
            self.search_count += 1
        elif self.search_count > 0:
            self.search_count -= 1
        if self.search_count == 3:
            return True
        elif self.search_count == 2 and self.state == 'searching':
            return True
        return False

    def searching_action(self, kart_loc, kart_front, action):
        kart_x = kart_loc[0]
        kart_y = kart_loc[1]
        perspective = np.sign(kart_front[0] - kart_loc[0])
        if kart_x < 0 and kart_y < 0:
            facing = (kart_front[1] - kart_loc[1]) > 0
            if facing:
                action['steer'] = .5
                action['drift'] = True
                action['acceleration'] = 1
            else:
                action['steer'] = -.5
                action['drift'] = True
                action['acceleration'] = .5
        elif kart_x < 0:
            facing = (kart_front[1] - kart_loc[1]) > 0
            if facing:
                action['steer'] = .5
                action['drift'] = True
                action['acceleration'] = 1
            else:
                action['steer'] = -.5
                action['drift'] = True
                action['acceleration'] = .5
        elif kart_x > 0 and kart_y < 0:
            facing = (kart_front[1] - kart_loc[1]) > 0
            if facing:
                action['steer'] = -.5
                action['drift'] = True
                action['acceleration'] = .5
            else:
                action['steer'] = .5
                action['drift'] = True
                action['acceleration'] = 1
        else:
            facing = (kart_front[1] - kart_loc[1]) > 0
            if facing:
                action['steer'] = -.5
                action['drift'] = True
                action['acceleration'] = .5
            else:
                action['steer'] = .5
                action['drift'] = True
                action['acceleration'] = 1
        if abs(self.kart_loc[0]) < 20:
            action['steer'] = perspective * action['steer']
        return action

    def attack_action(self, kart_loc, kart_front, puck_loc, action):
        vector_of_kart = self.get_vector_from_this_to_that(kart_loc, kart_front)
        vector_to_their_goal = self.get_vector_from_this_to_that(kart_loc, self.their_goal_center)
        facing_angle = math.degrees(self.angle_between(vector_of_kart, vector_to_their_goal))
        #vector_right = self.get_vector from_this_to_that(kart_loc, their_goal_right)
        vector_center = self.get_vector_from_this_to_that(kart_loc, their_goal_center)
        vector_left = self.get_vector_from_this_to_that(kart_loc, their_goal_left)
        attack_cone = math.degrees(self.angle_between(vector_left, vector_right))
        x = puck_loc[0]
        y = puck_loc[-1]
        action = {'acceleration': .5, 'steer': x, 'brake': False, 'drift': False, 'nitro': False}
        if x < 0.05 and x > -0.05:
            action['steer'] = x 
            action['acceleration'] = .5
        elif x > 0.05 and x < .2:
            action['steer'] = .25
            action['acceleration'] = .4
            action['brake'] = True
        elif x < -0.05 and x > -.2:
            action['steer'] = -.25
            action['acceleration'] = .4
            action['brake'] = True
        elif x > .2 and x < .4:
            action['steer'] = .75
            action['acceleration'] = .2
            action['brake'] = True
            action['drift'] = True
        elif x < -.2 and x > -.4:
            action['steer'] = -.75
            action['acceleration'] = .2
            action['brake'] = True
            action['drift'] = True
        elif x > 0.4 and x < 0.7:
            action['steer'] = 1
            action['acceleration'] = .1
            action['brake'] = True
            action['drift'] = True
        elif x < -0.4 and x > -0.7:
            action['steer'] = -1
            action['acceleration'] = .1
            action['brake'] = True
            action['drift'] = True
        elif x > 0.7:
            action['steer'] = 1
            action['acceleration'] = .3
            action['brake'] = True
            action['drift'] = False
        elif x < -0.7:
            action['steer'] = -1
            action['acceleration'] = .3
            action['brake'] = True
            action['drift'] = True
        if abs(x) < .2 and abs(y) < .2:
            if facing_angle < attack_cone / 2:
                action['steer'] = 0
                action['acceleration'] = 1
        return action

    @staticmethod
    def to_numpy(location):
        return np.float32([location[0], location[2]])

    @staticmethod
    def get_vector_from_this_to_that(me, obj, normalize=True):
        vector = obj - me
        if normalize:
            return vector / np.linalg.norm(vector)
        return vector

    def angle_between(self, v1, v2):
        return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

    def check_reset(self, kart_loc):
        threshold = 5
        last_loc = self.past_kart_locs[-1]
        x_diff = abs(last_loc[0] - kart_loc[0])
        y_diff = abs(last_loc[-1] - kart_loc[-1])
        if x_diff > threshold or y_diff > threshold:
            return True
        return False
