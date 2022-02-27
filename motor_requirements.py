import numpy as np
from scipy import constants


def w_from_rpm(rpm, translation=1.0):
    return 2 * np.pi * rpm / 60 / translation


def v_from_w(w, radius):
    return radius * w


def force_from_moment_and_radius(moment, radius, translation=1.0):
    return (moment * translation) / radius


def rpm_from_distance_and_time(distance, time, radius):
    return distance / ((radius * 2) * np.pi * time)


def torque_with_friction(radius, friction_factor, mass):
    return radius * friction_factor * mass * constants.g


def acceleration_from_distance_and_time(distance, time):
    return (distance * 2) / np.power(time, 2)


def torque_acceleration_from_distance_and_time(distance, time, radius, mass):
    return radius * mass * acceleration_from_distance_and_time(distance, time)


def motor_requirements(
        self,
        max_distance,
        max_time,
        buffer,
):
    max_time -= buffer

    min_rpm = rpm_from_distance_and_time(max_distance, max_time, self.wheel_radius)
    min_torque_friction = torque_with_friction(self.wheel_radius, self.roll_friction, self.car_mass)
    v = v_from_w(w_from_rpm(min_rpm), self.wheel_radius)
    min_acceleration = acceleration_from_distance_and_time(max_distance, max_time)
    min_torque_acceleration = min_acceleration * self.wheel_radius * self.car_mass

    return {
        'Min rpm - Rpm': min_rpm,
        'Min torque friction - mNm': min_torque_friction,
        'v - m/s': v,
        'min_acceleration - m/(s^2)': min_acceleration,
        'min_torque_acceleration - mNm': min_torque_acceleration,
        'sum_min_torque- Rpm': min_torque_friction + min_torque_acceleration,
        'sum_min_rpm - mNm': min_rpm,
    }