import genesis as gs
import torch
import os
from PIL import Image
import numpy as np

from genesis_forge import ManagedEnvironment, EnvMode
from genesis_forge.managers import (
    RewardManager,
    TerminationManager,
    EntityManager,
    ObservationManager,
    ActuatorManager,
    PositionActionManager,
    VelocityCommandManager,
    TerrainManager,
    ContactManager
)
from genesis_forge.mdp import reset, terminations, observations, rewards

HEIGHT_OFFSET = 0.09
INITIAL_BODY_POSITION = [0.0, 0.0, HEIGHT_OFFSET]
INITIAL_QUAT = [1.0, 0.0, 0.0, 0.0]

def pose_to_T(pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)):
    """Convierte posicion (x,y,z) y cuaternion (w,x,y,z) a matriz homogenea 4x4."""
    w, x, y, z = quat
    R = np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = pos
    return T


class DuckDuckEnv(ManagedEnvironment):
    def __init__(
        self,
        num_envs:             int        = 1,
        dt:                   float      = 1 / 50,
        max_episode_length_s: int | None = 20,
        headless:             bool       = True,
        mode: EnvMode = "train",
    ):
        super().__init__(
            num_envs=num_envs,
            dt=dt,
            max_episode_length_sec=max_episode_length_s,
            max_episode_random_scaling=0.1,
        )

        self._mode = mode
        self._curriculum_level = 0

        self.scene = gs.Scene(
            show_viewer=not headless,
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.5, 1.5, 1.0),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=45,
                refresh_rate= int(0.5/self.dt)
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt /2 ,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=False,
                max_collision_pairs=30,
            ),
            vis_options=gs.options.VisOptions(
                shadow=False,
                plane_reflection=False,
                rendered_envs_idx=list(range(1)),
            ),
        )

        self.terrain = self.create_terrain(self.scene) #type:ignore

        self.robot = self.scene.add_entity( #type:ignore
            gs.morphs.MJCF(
                file="model/robot.xml",
                pos=INITIAL_BODY_POSITION,
                quat=INITIAL_QUAT,
            ),
        )

        self.camera = self.scene.add_camera(
            pos=(1.5, 1.5, 1.0),
            lookat=(0.0, 0.0, 0.1),
            fov=45,
            res=(640, 480),
            env_idx=0,
            debug=True,
        )

        self.robot_camera = self.scene.add_camera(
            pos=(0.0, 0.0, 0.0),
            lookat=(0.0, 1.0, 0.0),
            fov=45,
            res=(640, 480),
            env_idx=0,
            debug=True,
        )

    def config(self):

        self.terrain_manager = TerrainManager(self)

        self.robot_manager = EntityManager(
            self,
            entity_attr="robot",
            on_reset={ 
                "position": { 
                    "fn": reset.randomize_terrain_position, #type:ignore
                    "params": {
                        "height_offset": HEIGHT_OFFSET,
                        "terrain_manager": self.terrain_manager,
                    },
                },
            },
        )

        self.actuator_manager = ActuatorManager(
            self,
            joint_names=[
                "left_hip.*", "left_knee", "left_ankle",
                "right_hip.*", "right_knee", "right_ankle",
                "neck_pitch", "head_pitch", "head_yaw", "head_roll"
            ],
            default_pos={
                "left_hip_yaw": 0.0,
                "left_hip_roll": -0.08726646259971647,
                "left_hip_pitch": -0.457924,
                "left_knee": -0.004940,
                "left_ankle": 0.452984,
                "neck_pitch": 0.3490658503988659,
                "head_pitch": 0.3490658503988659,
                "head_yaw": 0.0,
                "head_roll": 0.0,
                "right_hip_yaw": 0.0,
                "right_hip_roll": 0.08726646259971647,
                "right_hip_pitch": 0.457924,
                "right_knee": 0.004940,
                "right_ankle": -0.452984,
            },
            kp= 0.55,
            kv= 0.0
        )

        self.action_manager = PositionActionManager(
            self,
            scale= 0.5,
            use_default_offset=True,
            actuator_manager=self.actuator_manager,
        )

        self.velocity_command = VelocityCommandManager(
            self,
            range={ #type:ignore
                "lin_vel_x": [0.0, 2.0],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_z": [-0.5, 0.5],
            },
            resample_time_sec=5.0,
            debug_visualizer=True,
            debug_visualizer_cfg = { #type:ignore
                "envs_idx":[0],        
            },
        )

        self.feet_contact_manager = ContactManager(
            self,
            link_names=["ankle_left", "ankle_right"],
            track_air_time=True,
        )

        RewardManager(
            self,
            logging_enabled=True,
            cfg={ #type:ignore
                "is_alive": {
                    "weight": 0.5,
                    "fn": lambda env: rewards.is_alive(env),
                },
                "tracking_lin_vel": {
                    "weight": 1.5,
                    "fn": lambda env, r=self.robot_manager, v=self.velocity_command: (
                        rewards.command_tracking_lin_vel(env, entity_manager=r, vel_cmd_manager=v)
                    ),
                },
                "tracking_ang_vel": {
                    "weight": 0.5,
                    "fn": lambda env, r=self.robot_manager, v=self.velocity_command: (
                        rewards.command_tracking_ang_vel(env, entity_manager=r, vel_cmd_manager=v)
                    ),
                },
                "feet_air_time": {
                    "weight": 1.0,
                    "fn": lambda env, c=self.feet_contact_manager, v=self.velocity_command: (
                        rewards.feet_air_time(
                            env,
                            contact_manager=c,
                            time_threshold=0.2,
                            time_threshold_max=0.5,
                            vel_cmd_manager=v,
                        )
                    ),
                },
                "feet_slide": {
                    "weight": -0.2,
                    "fn": lambda env, c=self.feet_contact_manager: (
                        rewards.feet_slide(env, contact_manager=c)
                    ),
                },
                "ang_vel_xy": {
                    "weight": -0.3,
                    "fn": lambda env, r=self.robot_manager: (
                        rewards.ang_vel_xy_l2(env, entity_manager=r)
                    ),
                },
                "lin_vel_z": {
                    "weight": -0.2,
                    "fn": lambda env, r=self.robot_manager: (
                        rewards.lin_vel_z_l2(env, entity_manager=r)
                    ),
                },
                "flat_orientation": {
                    "weight": -0.5,
                    "fn": lambda env, r=self.robot_manager: (
                        rewards.flat_orientation_l2(env, entity_manager=r)
                    ),
                },
                "action_rate": {
                    "weight": -0.2,
                    "fn": lambda env: rewards.action_rate_l2(env),
                },
                "base_height": {
                    "weight": -0.5,
                    "fn": lambda env, t=self.terrain_manager, e=self.robot_manager: (
                        rewards.base_height(env, target_height=HEIGHT_OFFSET, terrain_manager=t, entity_manager=e)
                    ),
                },
                "stand_still": {
                    "weight": -0.5,
                    "fn": lambda env, a=self.actuator_manager, v=self.velocity_command: (
                        rewards.stand_still_joint_deviation_l1(env, vel_cmd_manager=v, actuator_manager=a)
                    ),
                },
                "contact_force": {
                    "weight": -0.01,
                    "fn": lambda env, c=self.feet_contact_manager: (
                        rewards.contact_force(env, contact_manager=c, threshold=1.0)
                    ),
                },
            },
        )

        self.termination_manager = TerminationManager(
            self,
            logging_enabled=True,
            term_cfg={ #type:ignore
                "timeout": {
                    "fn": terminations.timeout,
                    "time_out": True,
                },

                "out_of_bounds": {
                    "fn": terminations.out_of_bounds,
                    "params": {
                        "terrain_manager": self.terrain_manager,
                    },
                },

                "bad_orientation": {
                    "fn": terminations.bad_orientation,
                    "params": {
                        "limit_angle": 30.0,
                        "entity_manager": self.robot_manager,
                        "grace_steps": 20,
                    },
                },
            },
        )

        ObservationManager(
            self,
            name="policy",
            cfg={ #type:ignore
                "velocity_cmd": {"fn": self.velocity_command.observation},
                "angle_velocity": {
                    "fn": lambda env: self.robot_manager.get_angular_velocity(),
                },
                "linear_velocity": {
                    "fn": lambda env: self.robot_manager.get_linear_velocity(),
                },
                "projected_gravity": {
                    "fn": lambda env: self.robot_manager.get_projected_gravity(),
                },
                "actions": {
                    "fn": lambda env: self.action_manager.get_actions(),
                },
            },
        )

        ObservationManager(
            self,
            name="critic",
            cfg={ #type:ignore
                "velocity_cmd": {"fn": self.velocity_command.observation},
                "angle_velocity": {
                    "fn": lambda env: self.robot_manager.get_angular_velocity(),
                },
                "linear_velocity": {
                    "fn": lambda env: self.robot_manager.get_linear_velocity(),
                },
                "projected_gravity": {
                    "fn": lambda env: self.robot_manager.get_projected_gravity(),
                },
                "dof_position": {
                    "fn": lambda env: self.action_manager.get_dofs_position(),
                },
                "dof_velocity": {
                    "fn": lambda env: self.action_manager.get_dofs_velocity(),
                    "scale": 0.05,
                },
                "actions": {
                    "fn": lambda env: self.action_manager.get_actions(),
                },
                "dof_force": {
                    "fn": lambda env, a=self.actuator_manager: observations.entity_dofs_force(env, actuator_manager=a),
                    "scale": 0.01,
                },
                "contact_force": {
                    "fn": lambda env, c=self.feet_contact_manager: observations.contact_force(env, contact_manager=c),
                    "scale": 0.01,
                },
            },
        )
    
    def create_terrain(self, scene: gs.Scene):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        tile_path = os.path.join(this_dir, "checker.png")
        img = Image.open(tile_path)
        img = img.resize((128, 128), Image.Resampling.NEAREST) 
        checker_image = np.array(img)
        tiled_image = np.tile(checker_image, (24, 24, 1))

        return scene.add_entity(
            surface=gs.surfaces.Default(
                diffuse_texture=gs.textures.ImageTexture(
                    image_array=tiled_image,
                )
            ),
            morph=gs.morphs.Terrain(
                pos=(-12, -12, 0),
                n_subterrains=(1, 1),
                subterrain_size=(24, 24),
                vertical_scale=0.001,
                subterrain_types=[["flat_terrain"]],
                subterrain_parameters={
                    "random_uniform_terrain": {
                        "min_height": 0.0,
                        "max_height": 0.1,
                        "step": 0.05,
                        "downsampled_scale": 0.25,
                    },
                },
            ),
        )

    def build(self):
        super().build()
        self.camera.follow_entity(self.robot)

        cam_pos  = (0.0155, -9.13778e-05, -0.0733)
        cam_quat = (-0.7071068, 0, 0, 0.7071068) 
        offset_T = pose_to_T(pos=cam_pos, quat=cam_quat)
        cam_link = self.robot.get_link("jaw_soft")
        self.robot_camera.attach(cam_link, offset_T)
