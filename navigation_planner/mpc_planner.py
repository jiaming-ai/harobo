import numpy as np
import time

# Our planner
import casadi as ca
from .nmpc_casadi_oop import NMPC
# from harobo.navigation_planner.nmpc_casadi_oop import NMPC
from mapping.map_utils import MapSizeParameters

M_TO_CM = 100.0


def omni_dynamics(st, u, 
    params={
        "DT": .2, 
        # "L": np.sqrt(2) 
    }
):
    """ Dynamics of the omnidirectional 1st-order model (implemented using CasADi). 

    The discrete-time dynamic model is described as 

    .. math::

        \\mathbf x_{k+1} = A \\mathbf x_k + B \\mathbf u_k , 

    where 

    .. math:: 

        A = I_3 \\qquad
        B = \\begin{bmatrix}
            \\Delta t&         0&         0\\\\
            0        & \\Delta t&         0\\\\
            0        &         0& \\Delta t\\\\
        \\end{bmatrix}.


    Args: 
        st       : Current state. 
        u        : Control input. 
        params   : Other parameters. 

    Returns:
        CasADi row vector: Next state. 
    """
    DT = params["DT"] if "DT" in params else .2
    # L = params["L"] if "L" in params else np.sqrt(2)
    x_dot = u[0]
    y_dot = u[1]
    theta_dot = u[2]
    st_next = st + DT * ca.vertcat(*[x_dot, y_dot, theta_dot]).T 

    return st_next 


def omni_dynamics_2ord(st, u, 
    params={
        "DT": .2, 
    }
):
    """ Dynamics of the omnidirectional 2nd-order model (implemented using CasADi). 

    The discrete-time dynamic model is described as 
    
    .. math::

        \\mathbf x_{k+1} = A \\mathbf x_k + B \\mathbf u_k , 
    
    where 

    .. math::

        A = \\begin{bmatrix}
            1& 0& 0& \\Delta t& 0& 0\\\\
            0& 1& 0& 0& \\Delta t& 0\\\\
            0& 0& 1& 0& 0& \\Delta t\\\\
            0& 0& 0& 1& 0& 0\\\\
            0& 0& 0& 0& 1& 0\\\\
            0& 0& 0& 0& 0& 1 
        \\end{bmatrix}, \\quad
        B = \\begin{bmatrix}
            0.5\\Delta t^2&        0&         0\\\\
            0& 0.5\\Delta t^2&         0\\\\
            0&        0&  0.5\\Delta t^2\\\\
            \\Delta t&        0&         0\\\\
            0&       \\Delta t&         0\\\\
            0&        0&        \\Delta t\\\\
        \\end{bmatrix}


    Args: 
        st (CasADi row vector): Current state. 
        u (CasADi row vector): Control input. 
        params : Other parameters. 

    Returns:
        CasADi row vector: Next state. 
    """
    DT = params["DT"] if "DT" in params else .2
    NX = params["NX"] if "NX" in params else 6
    NU = params["NU"] if "NU" in params else 3
    assert NX == NU*2
    assert DT>0 
    A = np.vstack([np.hstack([np.zeros((NX//2, NX//2)), np.eye(NX//2)]), np.zeros((NX//2, NX))]) 
    A = DT * A + np.eye((NX))
    B = DT * np.vstack([0.5*DT*np.eye(NX//2, NU), np.eye(NU)]) 
    # A = np.array([
    #     [1, 0, 0, DT, 0, 0],
    #     [0, 1, 0, 0, DT, 0],
    #     [0, 0, 1, 0, 0, DT],
    #     [0, 0, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 1, 0],
    #     [0, 0, 0, 0, 0, 1],
    # ])
    # B = np.array([
    #     [0.5*DT^2,        0,         0],
    #     [       0, 0.5*DT^2,         0],
    #     [       0,        0,  0.5*DT^2],
    #     [      DT,        0,         0],
    #     [       0,       DT,         0],
    #     [       0,        0,        DT],
    # ])

    st_next = ca.mtimes([A, st.T]) + ca.mtimes([B, u.T]) 
    
    return st_next.T



class MPCController:
    """MPC Controller wrapper.  

    A wrapper for using CLF-CBF-NMPC motion planning. 
    This node 

        #. receives the target person's information passed down from upstream 
           perception modules, 
        #. updates its own goal pose (state), 
        #. formulates the control/planning problem as a (nonlinear) model 
           predictive control problem, 
        #. solves the numerical optimization, and finally 
        #. sends velocities (ROS Twists) to low-level control modules. 

    Example: 
        .. highlight:: python
        .. code-block:: python

            planner = MPCController()

            # Formulate the planning problem and set up the planner
            planner.setup_planner()

            # Loop
            while not planner.is_reach_goal():
                planner.plan(
                    state=x,
                    goal=np.array([5, 5, np.pi/2]),
                    obs=obstcl
                )

    Attributes: 
        is_sim (bool): True if running simulations; else False. 
        dyn_type (int): Robot's dynamic model: 1 for single-integrator, 
            2 for double integrators. 
        
        mpc_plate (:class:`numpc_casadi_oop.NMPC`): Plate for planner instance.
        mpc_instance (:class:`numpc_casadi_oop.NMPC`): Actual instance that runs 
            all optimizations of the planner. 
        distance (float): The desired robot-person distance [m]. 

        obstacle_map: (M, M) binary np.ndarray local obstacle map
            prediction.

        obstacles (list): obstacles positions. 
        detection_range (tuple or list): A range within which obstacles locate 
            are of interest. (min_dist, max_dist) [m]. 
        obs_sparsity_threshold (float): Obstacle sparsity threshold [m]. Two 
            LiDAR points lie within this distance are considered as one. 
        pose (:class:`numpy.array`): Robot's current **pose** (always zero in 
            the local frame). 
        goal (:class:`numpy.array`): Goal **state**. 
        planned_trj (:class:`numpy.ndarray`): Robot's planned **state** 
            trajectory. 
        ctrl_inputs (:class:`numpy.ndarray`): Robot's planned control input 
            trajectory. 
        plan_config (dict): Configurations for the NMPC planner. See 
            :class:`nmpc_casadi_oop.NMPC` for details. 
        plan_solver_config (dict): Settings for the NMPC planner's 
            optimization solver. See :class:`nmpc_casadi_oop.NMPC` for details.  
    """
    def __init__(self):
        self.is_sim = 1        
        self.dyn_type = 1
        if self.dyn_type not in [1, 2]:
            raise RuntimeError("Unknown dyanmics type!")
        
        self.mpc_plate = None
        self.mpc_instance = None

        # self.distance = rospy.get_param('~distance', 1.0)

        self.detection_range = [2e-1, 1.]
        self.obs_sparsity_threshold = 0.5

        self.plan_config = {
            "NT": 10,                   # horizon [1]
            "NX": 3,
            "NU": 3,
            "N_CBF": 2, 
            "DT": 0.2,                  # discretized time step [s] 
            "MAX_VEL": np.array([
                0.7,                # maximum x velocity [m/s]
                0.5,                # maximum y velocity [m/s]
                1.1,                # maximum angle velocity [rad/s]
            ]),
            "MAX_ACC": np.array([
                0.5,                # maximum x acceleration [m/s^2]
                0.3,                # maximum y acceleration [m/s^2]
                0.5,                # maximum angle acceleration [rad/s^2]
            ]),
            "P": np.diag(np.array([3.0, 3.0, 1.0])),
            "Q": np.diag(np.array([10.0, 10.0, 3.0])),
            "R": np.diag(np.array([0.001, 0.002, 0.001])),
            "R_d": np.diag(np.array([1e-6, 1e-6, 5e-6])),        # steer change cost
            "gamma_k": 0.3, 
            "tictoc": True, 
        }
        # self.plan_config = {
        #     "NT": rospy.get_param('~NT', 12),                   # horizon [1]
        #     "NX": rospy.get_param('~NX', 3),
        #     "NU": rospy.get_param('~NU', 3),
        #     "N_CBF": rospy.get_param('~N_CBF', 5), 
        #     "DT": rospy.get_param('~DT', 0.2),                  # discretized time step [s] 
        #     "MAX_VEL": np.array([
        #         rospy.get_param('~max_vx', 0.5),                # maximum x velocity [m/s]
        #         rospy.get_param('~max_vy', 0.3),                # maximum y velocity [m/s]
        #         rospy.get_param('~max_va', 0.5),                # maximum angle velocity [rad/s]
        #     ]),
        #     "MAX_ACC": np.array([
        #         rospy.get_param('~max_ax', 0.5),                # maximum x acceleration [m/s^2]
        #         rospy.get_param('~max_ay', 0.3),                # maximum y acceleration [m/s^2]
        #         rospy.get_param('~max_aa', 0.5),                # maximum angle acceleration [rad/s^2]
        #     ]),
        #     "P": np.diag(rospy.get_param('~P', np.array([50.0, 50.0, 70.0]))),
        #     "Q": np.diag(rospy.get_param('~Q', np.array([5.0, 5.0, 10.0]))),
        #     "R": np.diag(rospy.get_param('~R', np.array([0.01, 0.02, 0.001]))),
        #     "R_d": np.diag(rospy.get_param('~R_d', np.array([0.01, 0.02, 0.001]))),        # steer change cost
        #     "gamma_k": rospy.get_param('~gamma_k', 0.3), 
        #     "tictoc": rospy.get_param('~tictoc', True), 
        # }
        
        self.plan_solver_config = {
            'ipopt.max_iter':200, 
            'ipopt.print_level':0, 
            'print_time':0, 
            'ipopt.acceptable_tol':1e-3, 
            'ipopt.acceptable_obj_change_tol':1e-3,
        }
        
        self.pose = np.zeros(3)
        # self.vel = np.zeros(3)
        self.goal = np.zeros(self.plan_config["NX"])
        self.planned_trj = np.zeros(
            (self.plan_config["NT"]+1, self.plan_config["NX"])
        )
        self.ctrl_inputs = np.zeros(self.plan_config['NU']) 
        self.obstacles = []
        self.reach_goal_flg = False
        self.reach_goal_cnt = 0


    def _scan_cb(self, data):
        """ LiDAR points call back. 

        Receiving LiDAR points and update the points within the range of 
        interest to be obstacles. 
        """
        self.obstacles = []
        phi = data.angle_min 
        point_last = np.array([100, 100])
        for r in data.ranges:
            # Throw points that are far away or too close
            if (data.range_min <= r <= data.range_max) and (self.detection_range[0] <= r<=self.detection_range[1]):
                laser_point = np.array([
                    self.pose[0]+r*np.cos(phi+self.pose[2]), 
                    self.pose[1]+r*np.sin(phi+self.pose[2]) 
                ])
                if ( np.linalg.norm(laser_point-point_last) > self.obs_sparsity_threshold ):
                    self.obstacles.append( laser_point )
                    point_last = laser_point
                #end if
            #end if
            phi += data.angle_increment 
        # end for 
        self._draw_ob(self.obstacles)


    def _process_ob(self, pose, obstcl_map, map_params:MapSizeParameters = None):
        """Processing obstacles for further planning.

        Convert the binary obstacle map to a list of local coordinates wherein 
        obstacles are only within range of interest and sparse. 

        Args:
            pose (:class:`numpy.ndarray`): The local coordinate of the agent. 
            obstcl_map (:class:`numpy.ndarray`): The binary obstcle map where 
                1 for obstacle, 0 for free space. 
            map_params (:class:`MapSizeParameters`): Relevant parameters of 
                obstacle map; None for default, with type check. 

        Raises:
            AssertionError: Wrong map parameters provided. 
        """
        assert isinstance(map_params, MapSizeParameters), "Wrong map parameters!"
        p = map_params

        # print("Map size:", obstcl_map.shape)
        # print("Map parameters:", p)
        # print("Map resolution:", p.resolution)

        local_loc = (pose[:2] * M_TO_CM / p.resolution).astype(int) \
            + p.local_map_size //2 
        # print("local loc:", local_loc)
        detection_size = int(self.detection_range[1] * M_TO_CM / p.resolution)
        ## Check bounds 
        # print("detection_size:", detection_size)
        ## local_loc[i] - detection_size // 2 >= 0 
        detection_size = min(detection_size, min(2*local_loc))  # Not rigorous
        # print("detection_size:", detection_size)
        ## local_loc[i] + detection_size // 2 < obstcl_map.shape[i]
        detection_size = min(detection_size, min(2*(np.array(obstcl_map.shape)-local_loc)))  # Not rigorous
        # print("detection_size:", detection_size)
        (x1, y1) = (local_loc[0] - detection_size // 2, 
                    local_loc[1] - detection_size // 2)
        (x2, y2) = (x1 + detection_size, 
                    y1 + detection_size)
        roi = obstcl_map[x1:x2, y1:y2]              # region of interest 
        print("ROI size:", roi.shape)
        # print("ROI:", roi)
        ori_loc = local_loc - np.array([x1, y1])    # The origin's index in ROI
        obs_ind = np.argwhere(roi>0.1).tolist()

        ob_last = np.array([1e6, 1e6])
        obs = []
        for ob in obs_ind:
            # Get the obstacle's local coordinate (not map index)
            ob_coord = (np.array(ob) - ori_loc) * p.resolution / M_TO_CM 
            # CAUTION: For feasibility reason, filter out obstacles too close. 
            if np.linalg.norm(ob_coord) <= self.detection_range[0]:
                continue
            # Filter out obstacles beyond range or too dense
            if np.linalg.norm(ob_coord) <= self.detection_range[1] and \
                np.linalg.norm(ob_coord-ob_last) > self.obs_sparsity_threshold:
                obs.append(ob_coord)
                ob_last = ob_coord.copy()
            #end if
        #end for
        print("Obs List length:", len(obs))
        print("obstacles: ", obs)
        self.obstacles = obs


    def _reach_goal_once(self, tol=1e-1):
        return np.linalg.norm(self.pose-self.goal) <= tol


    def is_reach_goal(self, stable_time=.5):
        if self.reach_goal_cnt >= stable_time // self.plan_config['DT']:
            return True
        else:
            return False


    def setup_planner(self):
        self.mpc_plate = NMPC(params=self.plan_config)
        self.mpc_plate.add_dyn_constraints(dyn=omni_dynamics)
        self.mpc_plate.formulate_cost()
        self.mpc_plate.set_solver(self.plan_solver_config)


    def mpc_planning(self):
        """ The planning process. 

        Set up the planner and solve the optimization. 

        Args: 

        Raises:
            RuntimeWarning: Unknown dynamics type. 
        """
        if self.dyn_type == 1:
            self.mpc_instance.set_parameters(self.pose, self.goal) 
            # self.mpc_instance.warmstart(self.pose, self.ctrl_inputs[0])
        elif self.dyn_type == 2:
            self.mpc_instance.set_parameters(
                np.concatenate((self.pose, self.vel)), 
                self.goal
            ) 
            # self.mpc_instance.warmstart(np.concatenate((self.pose, self.vel)), self.ctrl_inputs[0]) 
        else:
            raise RuntimeWarning('Unknown dynamics type! Stop planning... ')
        #end ifelse
        flg_success = self.mpc_instance.solve()

        if not flg_success:
            # rospy.logwarn("Solver failed! Solutions' NOT reliable!") 
            self.vel = np.zeros(3)
            return 

        self.planned_trj = self.mpc_instance.x_reslt
        # print('x=',self.mpc_instance.x_reslt)
        self.ctrl_inputs = self.mpc_instance.u_reslt
        # print('u=',self.mpc_instance.u_reslt)
        if self.dyn_type == 1:
            self.vel = self.ctrl_inputs[0]
        elif self.dyn_type == 2:
            self.vel = self.planned_trj[1, 3:]  
        else:
            raise RuntimeWarning('Unknown dynamics type! Stoping the vehicle... ')


    def plan(self, state=None, goal=None, obs_map=None, map_params=None):
        """ Plan function. 

        Process updated local obtacles. 
        Plan and publish waypoints if success; stay put if not. 

        Args:
            state (:class:`numpy.ndarray`): Current state of the agent; 
                None if using local coordinate. 
            goal (:class:`numpy.ndarray`): Goal state; should be in the same 
                coordinate system as the current state. 
            obs_map (list): Local obstacle map; None for default. 
            map_params (:class:`MapSizeParameters`): Relevant parameters of 
                obstacle map; None for default. 
        
        Raises:
            AssertionError: Unknown goal.
        """
        if state is not None:
            self.pose = state.copy()
        else:
            self.pose = np.zeros(self.plan_config['NX'])
        assert goal is not None, "No goal provided!"
        self.goal = goal.copy()
        if self._reach_goal_once():
            self.reach_goal_flg = True
            self.reach_goal_cnt += 1
        else:
            self.reach_goal_flg = False
            self.reach_goal_cnt = 0
        
        tic = time.time()
        self._process_ob(
            pose=self.pose, 
            obstcl_map=obs_map, 
            map_params=map_params
        )
        toc = time.time()
        if self.mpc_plate.timing:
            print("Processing obstacle time: %.1f ms" \
                  % (1000*round(toc-tic, 4)))

        # Copy the plate (since obstacles are different now)
        self.mpc_instance = self.mpc_plate.copy()
        self.mpc_instance.add_obstacle_avoidance_constraints(self.obstacles)
        # Plan accordingly
        self.mpc_planning() 
        # TODO: Clip the velocity again for safety. 

        return self.planned_trj




######################################################################################
######################################## MAIN ########################################
######################################################################################
def main():
    """ Main for a simple test. 

    Caution: It's extremely slow using global coordination. 

    Raises:
        RuntimeError: Unknown dynamics type. 
    """

    # Obstacle
    map_param = MapSizeParameters(
        5, 
        500, 
        2
    )
    obstcl = np.zeros((100, 100))
    obstcl[30, 50] = 1
    obstcl[80, 10] = 1
    obstcl[20, 60] = 1
    for i in range(30, 81):
        obstcl[60, i] = 1
    # obstcl = [
    #     [3,2],
    #     [8,9],
    #     [1,5],
    #     [6,9],
    # ]
    goal = np.array([4.8, 4.3, np.pi/2])
    # Simple simulation
    x = np.zeros(3)
    dt = 0.1
    u = np.zeros(3)


    planner = MPCController()

    # Formulate the planning problem and set up the planner
    planner.setup_planner()

    # Loop
    while not planner.is_reach_goal(stable_time=.6):
        rel_goal = goal - x
        print("Rel. goal =", rel_goal)
        tic = time.time()
        planner.plan(
            state=x,
            goal=goal,
            # goal=rel_goal,
            obs_map=obstcl, 
            map_params=map_param
        )
        toc = time.time()
        print("Planning time: %.1f ms\n" % (1000*round(toc-tic, 4)) ) 
        u = planner.ctrl_inputs.copy()
        # print("Plan =", planner.planned_trj)
        print("u =", u[0])
        x = x + dt * u[0]
        if x[2] >= np.pi:
            x[2] -= 2*np.pi
        if x[2] < -np.pi:
            x[2] += 2*np.pi
        print("x =", x)


if __name__ == '__main__':
    main()
    # TODO: Relative coordination not working due to
    #  lack of dynamic local obs map implementation

