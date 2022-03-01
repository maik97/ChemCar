import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from scipy import interpolate


class ChemCarModel:

    def __init__(
            self,
            car_mass_total=10,
            alpha=0.5,
            additional_mass_factor=0.3,
            back_wheel_radius=0.075,
            back_wheel_mass=0.5,
            front_wheel_radius=0.075,
            front_wheel_mass=None,
            wheelbase=0.25,
            back_axis_mass_dist=0.5,
            roll_friction=0.01,
            translation=1,
            cw=0.3,
            car_surface=0.1,
            rho_air=1.1839,
            k=3,
            max_axis_moment=9999,
            motor_moment=0.392,
            motor_rpm=50,
    ) -> None:
        """
        Motor:
        https://www.reichelt.de/de/de/getriebemotor-66-3-mm-1-100-6-v-dc-gm66-3-6v-2-p270439.html?PROVID=2788&gclid=CjwKCAiApfeQBhAUEiwA7K_UHxUKspgjUi6MpxiKLCf33cVkxrLSjlAavW2S4wSYhpG7Lv3m6DJ_ExoCwMgQAvD_BwE&&r=1

        :param car_mass_total:
        :param alpha:
        :param additional_mass_factor:
        :param back_wheel_radius:
        :param back_wheel_mass:
        :param front_wheel_radius:
        :param front_wheel_mass:
        :param wheelbase:
        :param back_axis_mass_dist:
        :param roll_friction:
        :param translation:
        :param cw:
        :param car_surface:
        :param rho_air:
        :param k:
        :param max_axis_moment:
        :param motor_moment:
        :param motor_rpm:
        """

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
        #print(self.max_car_velocity)

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
            #print(axis_moment, car_acc, car_velocity)
            self.log_values(axis_moment, car_acc, car_velocity, dt)

        return self.values


def make_plot(x, y, xlabel, ylabel, title, path=None, fn='no_name'):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if path is None:
        plt.show()
    else:
        plt.savefig(path+"/"+fn+".svg")
    plt.clf()


def plot_results(val_dict, path=None):
    a_mom = np.array(val_dict['axis_moment'])*1000
    make_plot(val_dict['dt'], a_mom, "t in Sek", "M(t) in mNm", "Moment Hinterachse", path, fn='ax_mom')
    make_plot(val_dict['dt'], val_dict['car_acc'], "t in Sek", "a(t) in m/(s^2)", "Beschleunigung", path, fn='acc')
    make_plot(val_dict['dt'], val_dict['car_velocity'], "t in Sek", "v(t) in m/s", "Geschwindigkeit", path, fn='vel')
    make_plot(val_dict['dt'], val_dict['car_distance'], "t in Sek", "x(t) in m", "Strecke", path, fn='dist')


def time_from_dist(dist, val_dict, k=1):
    bspline = interpolate.make_interp_spline(val_dict['car_distance'], val_dict['dt'] , k=k)
    if isinstance(dist, list):
        return [bspline(elem) for elem in dist]
    else:
        return bspline(dist)


def test_different_weights(min_w=2, max_w=33, steps=35, *args, **kwargs):

    weight_results = {
        'car_mass' : [],
        'results_8' : [],
        'results_16' : []
    }

    for w in np.linspace(min_w, max_w, steps):

        print('test mass:', w)

        chem_car = ChemCarModel(car_mass_total=w, additional_mass_factor=0.0, *args, **kwargs)
        w_results = chem_car.test_motor()
        total_time_8, total_time_16 = time_from_dist([8, 16], w_results)

        weight_results['car_mass'].append(w)
        weight_results['results_8'].append(total_time_8)
        weight_results['results_16'].append(total_time_16)

    return weight_results


def main():

    path = 'results'

    w_r = test_different_weights()

    make_plot(w_r['car_mass'], w_r['results_8'],
              "Gewicht in kg", "Zeit in s",
              "Benötigte Zeit bis 8 Meter",
              path=path, fn="mass_8m"
              )

    make_plot(w_r['car_mass'], w_r['results_16'],
              "Gewicht in kg", "Zeit in s",
              "Benötigte Zeit bis 16 Meter",
              path=path, fn="mass_16m"
              )

    chem_car = ChemCarModel()
    vals = chem_car.test_motor()
    plot_results(vals, path=path)

if __name__ == '__main__':
    main()

