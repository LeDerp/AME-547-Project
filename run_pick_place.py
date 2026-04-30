import numpy as np
import mujoco
import mujoco.viewer


SCENE_XML = "universal_robots_ur10e/pick_place_scene.xml"

EE_BODY_CANDIDATES = [
    "vacuum_gripper",
    "tool0",
    "flange",
    "wrist_3_link",
    "ee_link",
]

EE_OFFSET = np.array([0.0, 0.0, 0.0])   # local offset from EE body origin to suction point
ARM_DOF = 6
STEPS_PER_WAYPOINT = 50


# ─────────────────────────────────────────────────────────────────────────────
# Rotation helpers
# ─────────────────────────────────────────────────────────────────────────────

def rotz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def roty(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])


def rotx(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0,  c, -s],
        [0.0,  s,  c],
    ])


def skew(v):
    x, y, z = v
    return np.array([
        [0.0, -z,   y],
        [z,    0.0, -x],
        [-y,   x,   0.0],
    ])


def orientation_error(R_current, R_target):
    """
    Small-angle orientation error in world coordinates.
    Returns a 3-vector that goes to zero when R_current == R_target.
    """
    return 0.5 * (
        np.cross(R_current[:, 0], R_target[:, 0]) +
        np.cross(R_current[:, 1], R_target[:, 1]) +
        np.cross(R_current[:, 2], R_target[:, 2])
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pose IK solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_ik_pose(
    model,
    ik_data,
    body_id,
    target_pos,
    target_rot,
    ee_offset,
    arm_dof=6,
    max_iters=1000,
    pos_tol=1e-3,
    rot_tol=2e-3,
    damping=1e-2,
    alpha=0.25,
    rot_weight=1.0,
):
    """
    Pose IK with damped least squares.

    Solves for joint angles so the end-effector reaches:
        - target_pos (world position)
        - target_rot (world orientation)

    Parameters
    ----------
    target_pos : (3,)
        Desired EE point position in world frame.
    target_rot : (3,3)
        Desired EE body rotation matrix in world frame.
    ee_offset : (3,)
        Local offset from EE body origin to actual tool contact point.
    rot_weight : float
        Relative weight of rotation error vs translation error.
    """
    target_pos = np.asarray(target_pos, dtype=float)
    target_rot = np.asarray(target_rot, dtype=float)

    for _ in range(max_iters):
        mujoco.mj_forward(model, ik_data)

        body_pos = ik_data.xpos[body_id].copy()
        body_rot = ik_data.xmat[body_id].reshape(3, 3).copy()

        p_offset_world = body_rot @ ee_offset
        cur_ee_pos = body_pos + p_offset_world

        pos_err = target_pos - cur_ee_pos
        rot_err = orientation_error(body_rot, target_rot)

        if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(rot_err) < rot_tol:
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, ik_data, jacp, jacr, body_id)

        # Jacobian of the offset contact point
        Jpos = jacp - skew(p_offset_world) @ jacr
        Jrot = jacr

        J = np.vstack([
            Jpos[:, :arm_dof],
            rot_weight * Jrot[:, :arm_dof],
        ])

        err = np.hstack([
            pos_err,
            rot_weight * rot_err,
        ])

        A = J @ J.T + damping**2 * np.eye(6)
        dq = J.T @ np.linalg.solve(A, err)

        ik_data.qpos[:arm_dof] = np.clip(
            ik_data.qpos[:arm_dof] + alpha * dq,
            model.jnt_range[:arm_dof, 0],
            model.jnt_range[:arm_dof, 1],
        )

    mujoco.mj_forward(model, ik_data)
    body_pos = ik_data.xpos[body_id].copy()
    body_rot = ik_data.xmat[body_id].reshape(3, 3).copy()
    pred_ee = body_pos + body_rot @ ee_offset

    pos_residual = np.linalg.norm(target_pos - pred_ee)
    rot_residual = np.linalg.norm(orientation_error(body_rot, target_rot))

    return (
        ik_data.qpos[:arm_dof].copy(),
        pred_ee,
        body_rot,
        pos_residual,
        rot_residual,
    )


# ─────────────────────────────────────────────────────────────────────────────
# UR10e Cartesian controller
# ─────────────────────────────────────────────────────────────────────────────

class UR10eCartesian:
    def __init__(self, model, data, viewer=None):
        self.model = model
        self.data = data
        self.viewer = viewer
        self.arm_dof = ARM_DOF
        self._ik = mujoco.MjData(model)

        body_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(model.nbody)
        ]
        ee_body = next((n for n in EE_BODY_CANDIDATES if n in body_names), None)
        if ee_body is None:
            raise ValueError(
                f"Could not find end-effector body. Tried: {EE_BODY_CANDIDATES}"
            )

        self.ee_body = ee_body
        self.ee_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.ee_body
        )

        print(f"[setup] EE body : {self.ee_body}")
        print(f"[setup] Arm DOF : {self.arm_dof}")

    def _step(self, n):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            if self.viewer and self.viewer.is_running():
                self.viewer.sync()

    def _get_ee_pos(self):
        mujoco.mj_forward(self.model, self.data)
        body_pos = self.data.xpos[self.ee_body_id].copy()
        body_rot = self.data.xmat[self.ee_body_id].reshape(3, 3).copy()
        return body_pos + body_rot @ EE_OFFSET

    def _get_ee_rot(self):
        mujoco.mj_forward(self.model, self.data)
        return self.data.xmat[self.ee_body_id].reshape(3, 3).copy()

    def _set_arm_target(self, q):
        self.data.ctrl[:self.arm_dof] = q

    def settle(self, steps=1000):
        self._step(steps)

    def vacuum_on(self, sheet_name):
        """
        Turn vacuum on (useful after descending until contact without sheet)
        """
        eq_name = f"vacuum_attach_{sheet_name}"
        eq_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            eq_name
        )
        self.data.eq_active[eq_id] = 1


    def vacuum_off(self, sheet_name):
        """
        Turn vacuum off (useful after descending until contact with sheet)
        """
        eq_name = f"vacuum_attach_{sheet_name}"
        eq_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            eq_name
        )
        self.data.eq_active[eq_id] = 0

    def is_touching(self, sheet_name):
        """
        Check if the end-effector is touching the specified sheet.
        """
        ee_body_id = self.ee_body_id
        sheet_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, sheet_name
        )

        for i in range(self.data.ncon):
            con = self.data.contact[i]

            body1 = self.model.geom_bodyid[con.geom1]
            body2 = self.model.geom_bodyid[con.geom2]

            if (body1 == ee_body_id and body2 == sheet_body_id) or \
                (body1 == sheet_body_id and body2 == ee_body_id):
                return True

        return False

    def move_to_pose(self, target_pos, target_rot, speed=0.05, rot_weight=1.0):
        """
        Move end-effector to a target position and orientation.

        target_pos : (3,)
        target_rot : (3,3)
        speed      : Cartesian speed in m/s
        rot_weight : larger = prioritize rotation more strongly
        """
        target_pos = np.asarray(target_pos, dtype=float)
        target_rot = np.asarray(target_rot, dtype=float)

        start_pos = self._get_ee_pos()
        total_dist = np.linalg.norm(target_pos - start_pos)

        if total_dist < 1e-5:
            total_dist = 0.0

        step_size = speed * self.model.opt.timestep * STEPS_PER_WAYPOINT
        n_waypoints = max(1, int(np.ceil(total_dist / step_size))) if total_dist > 0 else 1

        print(f"  Start     : {start_pos.round(4)}")
        print(f"  Target    : {target_pos.round(4)}")
        print(f"  Distance  : {total_dist:.4f} m")
        print(f"  Waypoints : {n_waypoints}")

        for i in range(1, n_waypoints + 1):
            a = i / n_waypoints
            waypoint = start_pos + a * (target_pos - start_pos)

            self._ik.qpos[:] = self.data.qpos[:]
            self._ik.qvel[:] = 0.0

            q_sol, _, _, _, _ = solve_ik_pose(
                self.model,
                self._ik,
                self.ee_body_id,
                waypoint,
                target_rot,
                EE_OFFSET,
                arm_dof=self.arm_dof,
                rot_weight=rot_weight,
            )

            self._set_arm_target(q_sol)
            self._step(STEPS_PER_WAYPOINT)

        final_pos = self._get_ee_pos()
        final_rot = self._get_ee_rot()
        pos_err = np.linalg.norm(final_pos - target_pos)
        rot_err = np.linalg.norm(orientation_error(final_rot, target_rot))

        print(f"  Final EE pos : {final_pos.round(4)}")
        print(f"  Pos err      : {pos_err*1000:.2f} mm")
        print(f"  Rot err      : {rot_err:.6f}\n")

    def move_to(self, target_pos, speed=0.05, rot_weight=1.0):
        """
        Keep current orientation, change only position.
        """
        current_rot = self._get_ee_rot()
        self.move_to_pose(target_pos, current_rot, speed=speed, rot_weight=rot_weight)

    def descend_until_contact(self, xy, start_z, sheet_name,
                          min_z=0.35, dz=0.002, settle_steps=80):
        """
        Descend vertically until contact is detected.
        """
        current_z = start_z
        descent_rot = self._get_ee_rot().copy()   # lock in the current orientation

        while current_z > min_z:
            current_z -= dz

            self._ik.qpos[:] = self.data.qpos[:]
            self._ik.qvel[:] = 0.0

            q_sol, _, _, _, _ = solve_ik_pose(
                self.model,
                self._ik,
                self.ee_body_id,
                np.array([xy[0], xy[1], current_z], dtype=float),
                descent_rot,
                EE_OFFSET,
                arm_dof=self.arm_dof,
                rot_weight=0.2
            )

            self._set_arm_target(q_sol)
            self._step(settle_steps)

            mujoco.mj_forward(self.model, self.data)

            if self.is_touching(sheet_name):
                return current_z

        return None

    def reposition_joints(self, joint1_delta, joint6_delta, steps=3000):
        """
        Reposition joints via orientation adjustment.
        """
        q_current = self.data.qpos[:self.arm_dof].copy()
        q_target = q_current.copy()

        # relative changes
        q_target[0] += joint1_delta
        q_target[5] += joint6_delta

        for i in range(steps):
            a = (i + 1) / steps
            a = 3*a**2 - 2*a**3   # smoothstep

            q_interp = q_current + a * (q_target - q_current)

            self.data.ctrl[:self.arm_dof] = q_interp
            self._step(1)

# ─────────────────────────────────────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────────────────────────────────────

def run():
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)

    # Initial UR10e joint pose
    HOME_Q = np.array([0.0, -1.57, -1.57, -1.57, 1.57, 0.0])

    data.qpos[:ARM_DOF] = HOME_Q
    data.ctrl[:ARM_DOF] = HOME_Q
    mujoco.mj_forward(model, data)

    # Sheet location from XML file
    SHEET_XY = np.array([0.8, -0.3])

    # Target location
    TARGET_XY = np.array([-0.3, 0.8])

    # Heights
    HOVER_Z   = 0.62

    # Keep vacuum head flat relative to table
    TABLE_PARALLEL_ROT0 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ])

    tilt = 0  # radians (0 degrees for now)

    Ry = np.array([
        [ np.cos(tilt), 0, np.sin(tilt)],
        [ 0,            1, 0],
        [-np.sin(tilt), 0, np.cos(tilt)],
    ])

    TILTED_ROT0 = Ry @ TABLE_PARALLEL_ROT0

    print("=" * 60)
    print("UR10e pose IK — move to sheet")
    print(f"Sheet XY    : {SHEET_XY}")
    print(f"Hover Z     : {HOVER_Z:.3f}")
    print("=" * 60)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 140
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.2
        viewer.cam.lookat[:] = [0.1, 0.3, 0.45]

        robot = UR10eCartesian(model, data, viewer=viewer)

        print("[1] Settle at home")
        robot.settle(1000)

        print("[2] Move above sheet")
        robot.move_to(
            [SHEET_XY[0], SHEET_XY[1], HOVER_Z],
            speed=0.08,
            rot_weight=1.5,
        )

        print("[3] Descend until contact")
        start_z = robot._get_ee_pos()[2]

        contact_z = robot.descend_until_contact(
            xy=SHEET_XY,
            start_z=start_z,
            sheet_name="sheet2",
            min_z=0.35,
            dz=0.002,
            settle_steps=80,
        )

        if contact_z is None:
            print("  No contact detected! Check setup.")
            return

        print(f"  Contact detected at z = {contact_z:.4f}")

        print("[4] Turn on vacuum")
        robot.vacuum_on("sheet2")

        print("[done] Robot is at the sheet, ready to begin picking.")

        print("[5] Move upwards")
        robot.move_to(
            [SHEET_XY[0], SHEET_XY[1], HOVER_Z],
            speed=0.02,
            rot_weight=1.5,
        )

        print("[6] Reposition by rotating to prevent collision")
        robot.reposition_joints(
            joint1_delta=np.pi / 2,
            joint6_delta=0,
            steps=10000,
        )

        print("[7] Move to target location with sheet")
        robot.move_to(
            [TARGET_XY[0], TARGET_XY[1], HOVER_Z],
            speed=0.02,
            rot_weight=1.5,
        )

        print("[8] Descend until contact with target")
        start_z = robot._get_ee_pos()[2]

        contact_z = robot.descend_until_contact(
            xy=TARGET_XY,
            start_z=start_z,
            sheet_name="table",
            min_z=0.35,
            dz=0.002,
            settle_steps=80,
        )

        print("[9] Turn off vacuum")
        robot.vacuum_off("sheet2")

        print("[10] Move upwards")
        robot.move_to(
            [TARGET_XY[0], TARGET_XY[1], HOVER_Z],
            speed=0.02,
            rot_weight=1.5,
        )

        print("[11] Rotate around back to tray")
        robot.reposition_joints(
            joint1_delta=-np.pi*3 / 5,
            joint6_delta=0,
            steps=6000,
        )

        print("[12] move to second sheet and adjust vacuum gripper")
        FINAL_ROT = rotz(-np.pi/2) @ TABLE_PARALLEL_ROT0

        robot.move_to_pose(
            [SHEET_XY[0], SHEET_XY[1], HOVER_Z],
            FINAL_ROT,
            speed=0.02,
            rot_weight=2.0,
        )

        print("[13] Rotate 90 degrees")
        robot.reposition_joints(
            joint1_delta=0,
            joint6_delta=np.pi/2,
            steps=10000,
        )

        print("[14] Descend until contact with sheet")
        start_z = robot._get_ee_pos()[2]

        contact_z = robot.descend_until_contact(
            xy=SHEET_XY,
            start_z=start_z,
            sheet_name="sheet1",
            min_z=0.35,
            dz=0.002,
            settle_steps=80,
        )

        if contact_z is None:
            print("  No contact detected! Check setup.")
            return

        print(f"  Contact detected at z = {contact_z:.4f}")

        print("[15] Turn on vacuum")
        robot.vacuum_on("sheet1")

        print("[16] Move upwards")
        robot.move_to(
            [SHEET_XY[0], SHEET_XY[1], HOVER_Z],
            speed=0.02,
            rot_weight=1.5,
        )

        print("[17] Reposition by rotating to prevent collision")
        robot.reposition_joints(
            joint1_delta=np.pi / 2,
            joint6_delta=0,
            steps=10000,
        )

        print("[18] Move to target location with sheet, higher clearance for 90 degree rotation")
        robot.move_to(
            [TARGET_XY[0], TARGET_XY[1], HOVER_Z],
            speed=0.02,
            rot_weight=2.0,
        )

        print("[19] Descend until contact with target")
        start_z = robot._get_ee_pos()[2]

        contact_z = robot.descend_until_contact(
            xy=TARGET_XY,
            start_z=start_z,
            sheet_name="table",
            min_z=0.35,
            dz=0.002,
            settle_steps=80,
        )

        if contact_z is None:
            print("  No contact detected! Check setup.")
            return

        print(f"  Contact detected at z = {contact_z:.4f}")


        print("[20] Turn off vacuum")
        robot.vacuum_off("sheet1")

        print("[21] Move upwards")
        robot.move_to(
            [TARGET_XY[0], TARGET_XY[1], HOVER_Z],
            speed=0.02,
            rot_weight=1.5,
        )

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    run()