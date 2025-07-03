from copy import deepcopy

class BallEventDetector():
    """
    Detects ball-related events such as successful passes and interceptions in a basketball game,
    based on changes in ball possession and team assignments.
    """
    def __init__(self):
        """
        Initializes the BallEventDetector.
        Currently, no parameters are required during initialization.
        """
        pass 

    def detect_passes(self,ball_acquisition,player_assignment):
        """
        Detects successful passes between players on the same team.

        A pass is considered successful if:
        - The ball possession changes from one player to another
        - Both players belong to the same team

        Args:
            ball_acquisition (list): A list of length `num_frames`, where each element
                represents the player_id who possesses the ball at that frame. -1 if no one.
            player_assignment (list): A list of dictionaries, one per frame.
                Each dictionary maps player_id to their team (1 or 2).

        Returns:
            list: A list of length `num_frames` where each element indicates if a pass occurred:
                -1: No pass  
                1: Team 1 completed a pass  
                2: Team 2 completed a pass
        """
        
        passes = [-1] * len(ball_acquisition)
        prev_holder=-1
        previous_frame=-1

        for frame in range(1, len(ball_acquisition)):
            if ball_acquisition[frame - 1] != -1:
                prev_holder = ball_acquisition[frame - 1]
                previous_frame= frame - 1
            
            current_holder = ball_acquisition[frame]
            
            if prev_holder != -1 and current_holder != -1 and prev_holder != current_holder:
                prev_team = player_assignment[previous_frame].get(prev_holder, -1)
                current_team = player_assignment[frame].get(current_holder, -1)

                if prev_team == current_team and prev_team != -1:
                    passes[frame] = prev_team

        return passes

    def detect_interceptions(self,ball_acquisition,player_assignment):
        """
        Detects interceptions when the ball possession switches between players on opposing teams.

        An interception is detected if:
        - The player in possession changes
        - The previous and current players belong to different teams

        Args:
            ball_acquisition (list): A list of length `num_frames`, where each element
                is the player_id with the ball. -1 if none.
            player_assignment (list): A list of dictionaries, one per frame.
                Each dictionary maps player_id to their assigned team.

        Returns:
            list: A list of length `num_frames` where each element indicates interception:
                -1: No interception  
                1: Team 1 intercepted  
                2: Team 2 intercepted
        """
        interceptions = [-1] * len(ball_acquisition)
        prev_holder=-1
        previous_frame=-1
        
        for frame in range(1, len(ball_acquisition)):
            if ball_acquisition[frame - 1] != -1:
                prev_holder = ball_acquisition[frame - 1]
                previous_frame= frame - 1

            current_holder = ball_acquisition[frame]
            
            if prev_holder != -1 and current_holder != -1 and prev_holder != current_holder:
                prev_team = player_assignment[previous_frame].get(prev_holder, -1)
                current_team = player_assignment[frame].get(current_holder, -1)
                
                if prev_team != current_team and prev_team != -1 and current_team != -1:
                    interceptions[frame] = current_team
        
        return interceptions