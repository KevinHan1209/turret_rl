"""Turret vs Drone Gymnasium Environment - Single Shot Version.

This module implements a 2D reinforcement learning environment where a stationary
turret must make a single decisive shot to intercept a drone flying across the
battlefield. The agent observes the drone's position and velocity, and decides
when and where to fire. Each episode allows only one shot, making this a
single-shot decision problem.
"""

import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg

from ..config.config import WorldConfig, RewardConfig


@dataclass
class Bullet:
    """Represents a bullet in the simulation."""
    position: np.ndarray  # Current position [x, y]
    velocity: np.ndarray  # Velocity vector [vx, vy]
    distance_traveled: float  # Total distance traveled
    active: bool  # Whether bullet is still in flight


class TurretEnv(gym.Env):
    """2D Turret vs Drone environment for single-shot reinforcement learning.

    The environment simulates a stationary turret at the origin that must
    make a single decisive shot to intercept a drone flying across the
    battlefield. The turret can rotate instantly and fire once per episode.

    Observation Space (4D):
        A continuous vector in turret-centric Cartesian coordinates:
        - x_d_norm: Drone x position / 150 (normalized to ~[-1, 1])
        - y_d_norm: Drone y position / 150 (normalized to ~[-1, 1])
        - vx_d_norm: Drone x velocity / v_max (normalized to ~[-1, 1])
        - vy_d_norm: Drone y velocity / v_max (normalized to ~[-1, 1])

        World bounds: [-150, 150] × [-150, 150] meters
        Turret is at origin (0, 0)

    Action Space (2D):
        A continuous vector controlling firing decision:
        - a[0]: Firing angle [-1, 1] mapped to azimuth [-π, π]
        - a[1]: Fire gate (>0 to fire, ≤0 to wait)
                Agent can fire at most once per episode

    Rewards:
        - +1.0 for hitting the drone (when firing)
        - -1.0 for missing (when firing but no valid intercept)
        - -1.0 if episode ends without firing (drone escapes/max steps)
        - -0.001 per timestep (encourages timely decision)
    """

    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 20
    }

    def __init__(
        self,
        world_config: Optional[WorldConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        render_mode: Optional[str] = None,
        max_bullets: int = 10
    ):
        """Initialize the Turret environment.

        Args:
            world_config: World configuration parameters
            reward_config: Reward configuration parameters
            render_mode: Rendering mode ('human', 'rgb_array', or None)
            max_bullets: Maximum number of bullets that can exist simultaneously
        """
        super().__init__()

        # Load configurations
        self.world_config = world_config or WorldConfig()
        self.reward_config = reward_config or RewardConfig()
        self.render_mode = render_mode
        self.max_bullets = max_bullets

        # Extract commonly used values
        self.dt = self.world_config.dt
        self.world_half_size = self.world_config.world_half_size

        # Define observation space: 4D continuous vector
        # [x_d_norm, y_d_norm, vx_d_norm, vy_d_norm]
        # All normalized to approximately [-1, 1] range
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        # Define action space: 2D continuous vector
        # a[0]: Firing angle (normalized) in range [-1, 1] -> maps to [-π, π]
        # a[1]: Fire gate (>0 to fire if shot not yet taken, ≤0 to wait)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # Initialize state variables
        self.drone_pos: Optional[np.ndarray] = None
        self.drone_vel: Optional[np.ndarray] = None
        self.drone_start: Optional[np.ndarray] = None
        self.drone_end: Optional[np.ndarray] = None
        self.bullets: List[Bullet] = []
        self.turret_angle: float = 0.0
        self.turret_cooldown_timer: float = 0.0
        self.step_count: int = 0

        # Single-shot tracking
        self.shot_taken: bool = False  # Flag to track if shot has been fired
        self.shot_result: Optional[str] = None  # 'hit', 'miss', or None

        # Rendering setup
        self.figure = None
        self.ax = None
        self.canvas = None

    def _sample_drone_trajectory(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Sample random boundary points ensuring the path passes near the origin.

        Returns:
            start_pos: Starting position on boundary
            end_pos: Ending position on boundary
            speed: Drone speed for this episode
        """
        max_attempts = 1000
        min_distance = self.world_config.min_approach_distance

        for _ in range(max_attempts):
            # Sample two different boundary points
            start_pos = self._sample_boundary_point()
            end_pos = self._sample_boundary_point()

            # Ensure points are different enough
            if np.linalg.norm(end_pos - start_pos) < 50.0:
                continue

            # Check if line passes within min_distance of origin
            # Using point-to-line-segment distance formula
            line_vec = end_pos - start_pos
            line_len = np.linalg.norm(line_vec)
            line_unit = line_vec / line_len

            # Project origin onto the line
            origin_to_start = -start_pos
            projection_length = np.dot(origin_to_start, line_unit)

            # Clamp projection to line segment
            projection_length = np.clip(projection_length, 0, line_len)
            closest_point = start_pos + projection_length * line_unit
            distance_to_origin = np.linalg.norm(closest_point)

            if distance_to_origin <= min_distance:
                # Valid trajectory found
                speed = np.random.uniform(
                    self.world_config.drone_speed_min,
                    self.world_config.drone_speed_max
                )
                return start_pos, end_pos, speed

        # Fallback: return a trajectory that definitely passes close
        # Start from top, go through a point near origin
        start_pos = np.array([0.0, self.world_half_size])
        end_pos = np.array([0.0, -self.world_half_size])
        speed = np.random.uniform(
            self.world_config.drone_speed_min,
            self.world_config.drone_speed_max
        )
        return start_pos, end_pos, speed

    def _sample_boundary_point(self) -> np.ndarray:
        """Sample a random point on the world boundary.

        Returns:
            A 2D point on the square boundary
        """
        # Choose which edge (0: top, 1: right, 2: bottom, 3: left)
        edge = np.random.randint(4)
        t = np.random.uniform(-1, 1)  # Parameter along the edge

        if edge == 0:  # Top edge
            return np.array([t * self.world_half_size, self.world_half_size])
        elif edge == 1:  # Right edge
            return np.array([self.world_half_size, t * self.world_half_size])
        elif edge == 2:  # Bottom edge
            return np.array([t * self.world_half_size, -self.world_half_size])
        else:  # Left edge
            return np.array([-self.world_half_size, t * self.world_half_size])

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset

        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        super().reset(seed=seed)

        # Sample drone trajectory
        self.drone_start, self.drone_end, drone_speed = self._sample_drone_trajectory()
        self.drone_pos = self.drone_start.copy()

        # Calculate drone velocity
        direction = self.drone_end - self.drone_start
        direction = direction / np.linalg.norm(direction)
        self.drone_vel = direction * drone_speed

        # Reset turret and bullets
        self.bullets = []
        self.turret_angle = 0.0
        self.turret_cooldown_timer = 0.0
        self.step_count = 0

        # Reset single-shot tracking
        self.shot_taken = False
        self.shot_result = None

        # Get initial observation
        obs = self._get_observation()

        info = {
            'drone_speed': drone_speed,
            'trajectory_length': np.linalg.norm(self.drone_end - self.drone_start)
        }

        return obs, info

    def _get_observation(self) -> np.ndarray:
        """Construct the 4D observation vector in turret-centric coordinates.

        Returns a normalized 4D vector containing:
            - x_d_norm: Drone x position / 150
            - y_d_norm: Drone y position / 150
            - vx_d_norm: Drone x velocity / v_max
            - vy_d_norm: Drone y velocity / v_max

        Returns:
            Normalized observation vector of shape (4,)
        """
        # Normalize drone position by world half-size (150 meters)
        x_d_norm = self.drone_pos[0] / self.world_half_size
        y_d_norm = self.drone_pos[1] / self.world_half_size

        # Normalize drone velocity by max possible speed
        v_max = self.world_config.drone_speed_max
        vx_d_norm = self.drone_vel[0] / v_max
        vy_d_norm = self.drone_vel[1] / v_max

        obs = np.array([x_d_norm, y_d_norm, vx_d_norm, vy_d_norm], dtype=np.float32)

        return obs

    def _will_hit_drone(
        self,
        p_d: np.ndarray,
        v_d: np.ndarray,
        firing_angle: float,
        v_b: float,
        drone_radius: float,
        max_bullet_range: float
    ) -> Tuple[bool, Optional[float]]:
        """Analytically determine if a shot will hit the drone.

        Solves the quadratic equation to find if there exists a time τ ≥ 0 where:
            || (p_d + v_d * τ) - (v_b * u * τ) || <= drone_radius

        Where:
            - p_d: Current drone position
            - v_d: Drone velocity vector
            - u: Bullet direction unit vector = [cos(angle), sin(angle)]
            - v_b: Bullet speed (scalar)

        Args:
            p_d: Drone position [x, y]
            v_d: Drone velocity [vx, vy]
            firing_angle: Azimuth angle in radians
            v_b: Bullet speed (m/s)
            drone_radius: Collision radius for drone
            max_bullet_range: Maximum range bullet can travel

        Returns:
            (hit, tau):
                - hit: True if bullet will intercept drone
                - tau: Time to impact in seconds (None if no hit)
        """
        # Bullet direction unit vector
        u = np.array([np.cos(firing_angle), np.sin(firing_angle)])

        # Relative velocity: w = v_d - v_b * u
        # This is the velocity of the drone in the bullet's reference frame
        w = v_d - v_b * u

        # Solve ||p_d + w * τ||² = r²
        # Expanding: (w·w) * τ² + 2 * (p_d·w) * τ + (p_d·p_d - r²) = 0
        # This is a quadratic equation: a*τ² + b*τ + c = 0

        a = np.dot(w, w)
        b = 2.0 * np.dot(p_d, w)
        c = np.dot(p_d, p_d) - drone_radius ** 2

        # Compute discriminant
        discriminant = b ** 2 - 4 * a * c

        # No real solutions means no intercept
        if discriminant < 0:
            return False, None

        # Solve for τ
        sqrt_discriminant = np.sqrt(discriminant)

        # Avoid division by zero
        if abs(a) < 1e-10:
            # Degenerate case: bullet and drone have same velocity in direction of fire
            # Check if drone is already within range
            if abs(c) < 1e-10:  # Already at collision distance
                return True, 0.0
            else:
                return False, None

        tau1 = (-b - sqrt_discriminant) / (2 * a)
        tau2 = (-b + sqrt_discriminant) / (2 * a)

        # We want the smallest non-negative τ (earliest intercept)
        valid_taus = [t for t in [tau1, tau2] if t >= 0]

        if not valid_taus:
            return False, None

        tau = min(valid_taus)

        # Check if intercept happens within bullet's max range
        bullet_travel_distance = v_b * tau
        if bullet_travel_distance > max_bullet_range:
            return False, None

        # Check if drone is still within world bounds at intercept time
        intercept_drone_pos = p_d + v_d * tau
        if (abs(intercept_drone_pos[0]) > self.world_half_size * 1.1 or
                abs(intercept_drone_pos[1]) > self.world_half_size * 1.1):
            return False, None

        return True, tau

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one timestep of the environment (single-shot version).

        The agent can fire at most once per episode. When the agent fires:
        1. The environment analytically determines if the shot will hit
        2. The episode immediately terminates with appropriate reward
        3. Bullet simulation is optional (for visualization only)

        If the agent never fires and the episode ends (drone escapes or max_steps),
        the episode terminates with a miss penalty.

        Args:
            action: 2D action vector
                    action[0]: Firing angle [-1, 1] mapped to [-π, π]
                    action[1]: Fire gate (>0 to fire if not already taken, ≤0 to wait)

        Returns:
            observation: New observation after the step
            reward: Reward for this step
            terminated: Whether episode ended (shot taken or drone escaped)
            truncated: Whether episode was truncated (max steps)
            info: Additional information
        """
        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False
        info = {'hit': False, 'shots_fired': 0, 'shot_result': None}

        # 1. Process turret action
        # Map action[0] from [-1, 1] to [-π, π]
        self.turret_angle = action[0] * np.pi

        # 2. Check if agent wants to fire (and hasn't fired yet)
        if action[1] > 0 and not self.shot_taken:
            # Agent is taking their one shot
            self.shot_taken = True
            info['shots_fired'] = 1

            # Analytically evaluate if this shot will hit
            will_hit, tau = self._will_hit_drone(
                p_d=self.drone_pos,
                v_d=self.drone_vel,
                firing_angle=self.turret_angle,
                v_b=self.world_config.bullet_speed,
                drone_radius=self.world_config.drone_radius,
                max_bullet_range=self.world_config.bullet_max_range
            )

            if will_hit:
                # Hit! Episode terminates with positive reward
                self.shot_result = 'hit'
                info['hit'] = True
                info['shot_result'] = 'hit'
                info['impact_time'] = tau
                reward += self.reward_config.hit_reward
                terminated = True

                # Optional: Create bullet for visualization purposes
                # (not used for RL, but can be rendered)
                bullet_dir = np.array([np.cos(self.turret_angle), np.sin(self.turret_angle)])
                bullet_vel = bullet_dir * self.world_config.bullet_speed
                new_bullet = Bullet(
                    position=np.array([0.0, 0.0]),
                    velocity=bullet_vel,
                    distance_traveled=0.0,
                    active=True
                )
                self.bullets.append(new_bullet)
            else:
                # Miss! Episode terminates with negative reward
                self.shot_result = 'miss'
                info['shot_result'] = 'miss'
                reward += self.reward_config.miss_penalty
                terminated = True

                # Optional: Create bullet for visualization purposes
                bullet_dir = np.array([np.cos(self.turret_angle), np.sin(self.turret_angle)])
                bullet_vel = bullet_dir * self.world_config.bullet_speed
                new_bullet = Bullet(
                    position=np.array([0.0, 0.0]),
                    velocity=bullet_vel,
                    distance_traveled=0.0,
                    active=True
                )
                self.bullets.append(new_bullet)

        # 3. Update simulation state (even if terminated, for visualization)
        # Update drone position
        self.drone_pos = self.drone_pos + self.drone_vel * self.dt

        # Update any bullets (for visualization)
        for bullet in self.bullets:
            if bullet.active:
                bullet.position = bullet.position + bullet.velocity * self.dt
                bullet.distance_traveled += np.linalg.norm(bullet.velocity) * self.dt
                if bullet.distance_traveled >= self.world_config.bullet_max_range:
                    bullet.active = False

        # 4. If shot not taken, continue episode logic
        if not terminated:
            # Apply small step penalty (encourages timely decision)
            reward += self.reward_config.step_penalty

            # Check if drone escaped (moved past boundary)
            if (abs(self.drone_pos[0]) > self.world_half_size * 1.1 or
                    abs(self.drone_pos[1]) > self.world_half_size * 1.1):
                # Episode ends - no shot taken, drone escaped
                if not self.shot_taken:
                    # Use no_shot_penalty if available, otherwise miss_penalty
                    penalty = getattr(self.reward_config, 'no_shot_penalty', self.reward_config.miss_penalty)
                    reward += penalty
                    info['shot_result'] = 'no_shot_escaped'
                terminated = True

            # Check if max steps reached
            if self.step_count >= self.world_config.max_steps:
                # Episode truncated - no shot taken in time
                if not self.shot_taken:
                    # Use no_shot_penalty if available, otherwise miss_penalty
                    penalty = getattr(self.reward_config, 'no_shot_penalty', self.reward_config.miss_penalty)
                    reward += penalty
                    info['shot_result'] = 'no_shot_timeout'
                truncated = True

        # 5. Get new observation
        obs = self._get_observation()

        # Add additional info
        info.update({
            'step': self.step_count,
            'drone_distance': np.linalg.norm(self.drone_pos),
            'active_bullets': sum(1 for b in self.bullets if b.active),
            'turret_angle': self.turret_angle,
            'shot_taken': self.shot_taken
        })

        return obs, reward, terminated, truncated, info

    def _check_segment_sphere_intersection(
        self,
        seg_start: np.ndarray,
        seg_end: np.ndarray,
        sphere_center: np.ndarray,
        sphere_radius: float
    ) -> bool:
        """Check if a line segment intersects with a sphere.

        This provides more accurate collision detection for fast-moving bullets.

        Args:
            seg_start: Start point of line segment
            seg_end: End point of line segment
            sphere_center: Center of sphere
            sphere_radius: Radius of sphere

        Returns:
            True if segment intersects sphere
        """
        # Vector from start to end of segment
        seg_vec = seg_end - seg_start
        seg_length = np.linalg.norm(seg_vec)

        if seg_length == 0:
            # Degenerate segment, just check point distance
            return np.linalg.norm(seg_start - sphere_center) <= sphere_radius

        seg_unit = seg_vec / seg_length

        # Vector from segment start to sphere center
        to_center = sphere_center - seg_start

        # Project sphere center onto line
        projection = np.dot(to_center, seg_unit)

        # Clamp projection to segment
        projection = np.clip(projection, 0, seg_length)

        # Find closest point on segment
        closest_point = seg_start + projection * seg_unit

        # Check distance from closest point to sphere center
        distance = np.linalg.norm(closest_point - sphere_center)

        return distance <= sphere_radius

    def render(self) -> Optional[np.ndarray]:
        """Render the current state of the environment.

        Returns:
            RGB array if render_mode is 'rgb_array', None otherwise
        """
        if self.render_mode is None:
            return None

        # Create figure if it doesn't exist
        if self.figure is None:
            self.figure = plt.figure(figsize=(8, 8))
            self.ax = self.figure.add_subplot(111)
            if self.render_mode == 'rgb_array':
                self.canvas = FigureCanvasAgg(self.figure)

        # Clear the plot
        self.ax.clear()

        # Set axis properties
        self.ax.set_xlim(-self.world_half_size * 1.1, self.world_half_size * 1.1)
        self.ax.set_ylim(-self.world_half_size * 1.1, self.world_half_size * 1.1)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title(f'Turret vs Drone - Step {self.step_count}')

        # Draw world boundary
        boundary = patches.Rectangle(
            (-self.world_half_size, -self.world_half_size),
            self.world_config.world_size,
            self.world_config.world_size,
            linewidth=2,
            edgecolor='black',
            facecolor='none'
        )
        self.ax.add_patch(boundary)

        # Draw drone trajectory (faint line from start to end)
        if self.drone_start is not None and self.drone_end is not None:
            self.ax.plot(
                [self.drone_start[0], self.drone_end[0]],
                [self.drone_start[1], self.drone_end[1]],
                'g--',
                alpha=0.3,
                linewidth=1,
                label='Drone Path'
            )

        # Draw turret at origin
        turret_circle = patches.Circle(
            (0, 0),
            self.world_config.turret_radius,
            color='blue',
            label='Turret'
        )
        self.ax.add_patch(turret_circle)

        # Draw turret direction
        turret_line_end = np.array([
            np.cos(self.turret_angle) * self.world_config.turret_radius * 2,
            np.sin(self.turret_angle) * self.world_config.turret_radius * 2
        ])
        self.ax.plot([0, turret_line_end[0]], [0, turret_line_end[1]], 'b-', linewidth=2)

        # Draw drone
        if self.drone_pos is not None:
            drone_circle = patches.Circle(
                self.drone_pos,
                self.world_config.drone_radius,
                color='red',
                label='Drone'
            )
            self.ax.add_patch(drone_circle)

            # Draw velocity vector
            vel_scale = 0.3  # Scale factor for visualization
            vel_end = self.drone_pos + self.drone_vel * vel_scale
            self.ax.arrow(
                self.drone_pos[0],
                self.drone_pos[1],
                self.drone_vel[0] * vel_scale,
                self.drone_vel[1] * vel_scale,
                head_width=2,
                head_length=1,
                fc='red',
                ec='red',
                alpha=0.7
            )

        # Draw bullets
        for bullet in self.bullets:
            if bullet.active:
                bullet_circle = patches.Circle(
                    bullet.position,
                    self.world_config.bullet_radius,
                    color='orange',
                    zorder=10  # Draw on top
                )
                self.ax.add_patch(bullet_circle)

        # Add legend
        self.ax.legend(loc='upper right')

        # Add info text
        info_text = f"Active Bullets: {sum(1 for b in self.bullets if b.active)}\n"
        info_text += f"Drone Speed: {np.linalg.norm(self.drone_vel):.1f} m/s\n"
        info_text += f"Distance to Drone: {np.linalg.norm(self.drone_pos):.1f} m"
        self.ax.text(
            0.02, 0.98, info_text,
            transform=self.ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        if self.render_mode == 'human':
            plt.pause(0.001)
            return None
        else:  # rgb_array
            # Force matplotlib to fully redraw
            self.figure.canvas.draw()

            # Get the RGBA buffer
            buf = self.figure.canvas.buffer_rgba()
            img = np.frombuffer(buf, dtype=np.uint8).copy()  # .copy() to ensure new memory
            img = img.reshape(self.figure.canvas.get_width_height()[::-1] + (4,))
            return img[:, :, :3]  # Remove alpha channel

    def simulate_post_shot(self, n_steps: int = 40) -> List[np.ndarray]:
        """Continue simulating for visualization after episode ends.

        This is useful for video recording - after the agent takes their shot
        and the episode logically terminates, we can continue simulating the
        bullet and drone movement to create better visualizations.

        Args:
            n_steps: Number of additional simulation steps to run

        Returns:
            List of rendered frames
        """
        frames = []

        for i in range(n_steps):
            # Update drone position
            if self.drone_pos is not None:
                self.drone_pos = self.drone_pos + self.drone_vel * self.dt

            # Update bullets
            for bullet in self.bullets:
                if bullet.active:
                    bullet.position = bullet.position + bullet.velocity * self.dt
                    bullet.distance_traveled += np.linalg.norm(bullet.velocity) * self.dt
                    if bullet.distance_traveled >= self.world_config.bullet_max_range:
                        bullet.active = False

            # Increment step count for visualization
            self.step_count += 1

            # Render frame
            frame = self.render()
            if frame is not None:
                frames.append(frame)

        return frames

    def close(self):
        """Clean up resources."""
        if self.figure is not None:
            plt.close(self.figure)
            self.figure = None
            self.ax = None
            self.canvas = None