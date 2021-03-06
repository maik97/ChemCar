import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from scipy import interpolate


class ChemCarModel:

    def __init__(
            self,
            car_mass_total: float = 20,
            alpha = 0.5,
            additional_mass_factor: float = 0.3,
            back_wheel_radius: float = 0.1,
            back_wheel_mass: float = 0.5,
            front_wheel_radius: float = 0.07,
            front_wheel_mass: float = None,
            wheelbase: float = 0.465,
            back_axis_mass_dist: float = 0.5,
            roll_friction: float = 0.01,
            translation:float = 3,
            cw=0.3,
            car_surface=0.1,
            rho_air=1.1839,
            k=3,
            max_axis_moment=0.2,
            motor_moment=0.3,
            motor_rpm=55,
    ) -> None:

        car_mass = car_mass_total * (1 + additional_mass_factor)

        back_wheel_volume = 2 * np.power(np.pi, 2) * np.power(back_wheel_radius, 3)


        front_wheel_volume = 2 * np.power(np.pi, 2) * np.power(front_wheel_radius, 3)
        if front_wheel_mass is None:
            front_wheel_mass = back_wheel_mass * front_wheel_volume / back_wheel_volume

        back_axis_mass = back_axis_mass_dist * car_mass
        front_axis_mass = (1 - back_axis_mass_dist) * car_mass
        beta = np.arctan((back_wheel_radius - front_wheel_radius) / wheelbase)

        back_wheel_inertia_torque = back_wheel_mass * np.power(back_wheel_radius, 2)
        front_wheel_inertia_torque = front_wheel_mass * np.power(front_wheel_radius, 2)

        A_res_Luft = 0.5 * car_surface * cw * rho_air
        f_extra = 0.5 * car_mass * constants.g * roll_friction

        acc_devision = (
            car_mass * (k + 1) * (
                alpha
                + (1 - alpha) / (1 - np.tan(beta) * roll_friction)
                + 2 * back_wheel_inertia_torque / np.power(back_wheel_radius, 2)
                )
            )

        self.acc_func_factor = (translation / back_wheel_radius) / acc_devision

        self.acc_func_air_factor = (- 0.5 * (1 + (1 / (1 - np.tan(beta) * roll_friction))) * A_res_Luft) / acc_devision

        self.acc_func_rest = ((
            - (front_axis_mass * constants.g * roll_friction) / (1 - np.tan(beta) * roll_friction)
            - f_extra / 2
            ) / acc_devision
        )

        self.motor_moment = motor_moment
        self.max_axis_moment = max_axis_moment
        self.max_car_velocity = back_wheel_radius * 2 * np.pi * motor_rpm / 60 / translation
        print(self.max_car_velocity)

        self.back_wheel_inertia_torque = back_wheel_inertia_torque
        self.back_wheel_radius = back_wheel_radius

    def acceleration_function(self, motor_moment, car_velocity):
        return motor_moment * self.acc_func_factor \
               + np.power(car_velocity, 2) * self.acc_func_air_factor \
               + self.acc_func_rest

    def step(self, dt, motor_moment, prev_car_velocity):
        car_acc = self.acceleration_function(motor_moment, prev_car_velocity)
        axis_moment = 2 * self.back_wheel_inertia_torque * car_acc / self.back_wheel_radius
        axis_moment = np.clip(axis_moment, 0, self.max_axis_moment)
        car_acc = axis_moment / (2 * self.back_wheel_inertia_torque) * self.back_wheel_radius
        car_velocity = prev_car_velocity + car_acc * dt

        if car_velocity > self.max_car_velocity:
            car_velocity = self.max_car_velocity
            car_acc = (car_velocity - prev_car_velocity) / dt

        return axis_moment, car_acc, car_velocity

    def reset(self):
        self.values = {
            'dt': [0.0],
            'axis_moment': [0.0],
            'car_acc': [0.0],
            'car_velocity': [0.0],
            'car_distance': [0.0],
        }

    def log_values(self, axis_moment, car_acc, car_velocity, dt):
        self.values['dt'].append(self.values['dt'][-1] + dt)
        self.values['axis_moment'].append(axis_moment)
        self.values['car_acc'].append(car_acc)
        self.values['car_velocity'].append(car_velocity)
        self.values['car_distance'].append(self.values['car_distance'][-1] + car_velocity * dt)

    def test_motor(self, dt=0.01, distance=16):
        self.reset()

        car_velocity = 0.0
        while self.values['car_distance'][-1] < distance:
            axis_moment, car_acc, car_velocity = self.step(dt, self.motor_moment, car_velocity)
            print(axis_moment, car_acc, car_velocity)
            self.log_values(axis_moment, car_acc, car_velocity, dt)

        return self.values


def plot_results(val_dict):
    print(len(val_dict['dt']))
    plt.plot(val_dict['dt'], val_dict['axis_moment'])
    plt.show()
    plt.plot(val_dict['dt'], val_dict['car_acc'])
    plt.show()
    plt.plot(val_dict['dt'], val_dict['car_velocity'])
    plt.show()
    plt.plot(val_dict['dt'], val_dict['car_distance'])
    plt.show()


def time_from_dist(dist, val_dict, k=1):
    bspline = interpolate.make_interp_spline(val_dict['dt'], val_dict['car_distance'], k=k)
    if isinstance(dist, list):
        return [bspline(elem) for elem in dist]
    else:
        return bspline(dist)


def test_different_weights(min_w, max_w, steps, *args, **kwargs):

    for w in np.linspace(min_w, max_w, steps):
        chem_car = ChemCarModel(car_mass_total=w, *args, **kwargs)



def main():

    chem_car = ChemCarModel(
        car_mass_total = 10,
        alpha = 0.5,
        additional_mass_factor = 0.3,
        back_wheel_radius = 0.1,
        back_wheel_mass = 0.5,
        front_wheel_radius = 0.1,
        front_wheel_mass = None,
        wheelbase = 0.465,
        back_axis_mass_dist = 0.5,
        roll_friction = 0.01,
        translation = 3,
        cw = 0.3,
        car_surface = 0.1,
        rho_air = 1.1839,
        k = 3,
        max_axis_moment = 9999,
        motor_moment = 0.3
    )
    #chem_car = ChemCarModel()
    vals = chem_car.test_motor()
    plot_results(vals)

if __name__ == '__main__':
    main()

