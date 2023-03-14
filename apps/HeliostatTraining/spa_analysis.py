import matplotlib.pyplot as plt
import datetime
import math
from typing import Tuple

###############
#   DLR SPA   #
###############
def calculateSunAngles(
        hour: int,
        minute: int,
        sec: int,
        day: int,
        month: int,
        year: int,
        observerLatitude: float,
        observerLongitude: float,
) -> Tuple[float, float]:
    # in- and outputs are in degree
    if (
            hour < 0 or hour > 23
            or minute < 0 or minute > 59
            or sec < 0 or sec > 59
            or day < 1 or day > 31
            or month < 1 or month > 12
    ):
        raise ValueError(
            "at least one value exeeded time range in calculateSunAngles")

    else:
        observerLatitudeInt = observerLatitude / 180.0 * math.pi
        observerLongitudeInt = observerLongitude / 180.0 * math.pi

        pressureInput = 1.01325  # Pressure in bar
        temperature = 20  # Temperature in °C

        UT = hour + minute / 60.0 + sec / 3600.0
        pressure = pressureInput / 1.01325
        delta_t = 0.0

        if month <= 2:
            dyear = year - 1.0
            dmonth = month + 12.0
        else:
            dyear = year
            dmonth = month

        trunc1 = math.floor(365.25 * (dyear - 2000))
        trunc2 = math.floor(30.6001 * (dmonth + 1))
        JD_t = trunc1 + trunc2 + day + UT / 24.0 - 1158.5
        t = JD_t + delta_t / 86400.0

        # standard JD and JDE
        # (useless for the computation, they are computed for completeness)
        # JDE = t + 2452640
        # JD = JD_t + 2452640

        # HELIOCENTRIC LONGITUDE
        # linear increase + annual harmonic
        ang = 0.0172019 * t - 0.0563
        heliocLongitude = (
            1.740940
            + 0.017202768683 * t
            + 0.0334118 * math.sin(ang)
            + 0.0003488 * math.sin(2.0 * ang)
        )

        # Moon perturbation
        heliocLongitude = \
            heliocLongitude + 0.0000313 * math.sin(0.2127730 * t - 0.585)
        # Harmonic correction
        heliocLongitude = (
            heliocLongitude
            + 0.0000126 * math.sin(0.004243 * t + 1.46)
            + 0.0000235 * math.sin(0.010727 * t + 0.72)
            + 0.0000276 * math.sin(0.015799 * t + 2.35)
            + 0.0000275 * math.sin(0.021551 * t - 1.98)
            + 0.0000126 * math.sin(0.031490 * t - 0.80)
        )

        # END HELIOCENTRIC LONGITUDE CALCULATION
        # Correction to longitude due to notation
        t2 = t / 1000.0
        heliocLongitude = (
            heliocLongitude
            + (
                (
                    (-0.000000230796 * t2 + 0.0000037976) * t2
                    - 0.000020458
                ) * t2
                + 0.00003976
            ) * t2 * t2
        )

        delta_psi = 0.0000833 * math.sin(0.0009252 * t - 1.173)

        # Earth axis inclination
        epsilon = (
            -0.00000000621 * t
            + 0.409086
            + 0.0000446 * math.sin(0.0009252 * t + 0.397)
        )
        # Geocentric global solar coordinates
        geocSolarLongitude = heliocLongitude + math.pi + delta_psi - 0.00009932

        s_lambda = math.sin(geocSolarLongitude)
        rightAscension = math.atan2(
            s_lambda * math.cos(epsilon),
            math.cos(geocSolarLongitude),
        )

        declination = math.asin(math.sin(epsilon) * s_lambda)

        # local hour angle of the sun
        hourAngle = (
            6.30038809903 * JD_t
            + 4.8824623
            + delta_psi * 0.9174
            + observerLongitudeInt
            - rightAscension
        )

        c_lat = math.cos(observerLatitudeInt)
        s_lat = math.sin(observerLatitudeInt)
        c_H = math.cos(hourAngle)
        s_H = math.sin(hourAngle)

        # Parallax correction to Right Ascension
        d_alpha = -0.0000426 * c_lat * s_H
        # topOCRightAscension = rightAscension + d_alpha
        # topOCHourAngle = hourAngle - d_alpha

        # Parallax correction to Declination
        topOCDeclination = \
            declination - 0.0000426 * (s_lat - declination * c_lat)

        s_delta_corr = math.sin(topOCDeclination)
        c_delta_corr = math.cos(topOCDeclination)
        c_H_corr = c_H + d_alpha * s_H
        s_H_corr = s_H - d_alpha * c_H

        # Solar elevation angle, without refraction correction
        elevation_no_refrac = math.asin(
            s_lat * s_delta_corr
            + c_lat * c_delta_corr * c_H_corr
        )

        # Refraction correction:
        # it is calculated only if elevation_no_refrac > elev_min
        elev_min = -0.01

        if elevation_no_refrac > elev_min:
            refractionCorrection = (
                0.084217 * pressure
                / (273.0 + temperature)
                / math.tan(
                    elevation_no_refrac
                    + 0.0031376 / (elevation_no_refrac + 0.089186)
                )
            )
        else:
            refractionCorrection = 0

        # elevationAngle = \
        #     np.pi / 2 - elevation_no_refrac - refractionCorrection
        elevationAngle = elevation_no_refrac + refractionCorrection
        elevationAngle = elevationAngle * 180 / math.pi

        # azimuthAngle = math.atan2(
        #     s_H_corr,
        #     c_H_corr * s_lat - s_delta_corr/c_delta_corr * c_lat,
        # )
        azimuthAngle = -math.atan2(
            s_H_corr,
            c_H_corr * s_lat - s_delta_corr/c_delta_corr * c_lat,
        )
        azimuthAngle = azimuthAngle * 180 / math.pi

    return azimuthAngle, elevationAngle

##################
#   Experiment   #
##################
class Result:
    def __init__(self, 
                 azimuth : float,
                 elevation : float,
                 azim_velocity : float,
                 elev_velocity : float,
                ):
        self.azimuth = azimuth
        self.elevation = elevation
        self.azim_velocity = azim_velocity
        self.elev_velocity = elev_velocity
        self.velocity = math.sqrt(self.azim_velocity**2 + self.elev_velocity**2)

def main():
    start_date : datetime.datetime = datetime.datetime(year=2022, month=1, day=1, hour=8)
    end_date : datetime.datetime = datetime.datetime(year=2023, month=1, day=1, hour=1)
    min_hour = 7
    max_hour = 18

    #############
    #   Dates   #
    #############
    dates = [start_date]
    last_date = start_date
    while dates[-1] <= end_date:

        # add new date every 10 minutes
        new_date = last_date + datetime.timedelta(minutes=10)
        last_date = new_date

        # skip dates outside hour range
        if new_date.hour >= min_hour and new_date.hour <= max_hour:
            dates.append(new_date)

    ####################
    #   Solar Angles   #
    ####################
    results = {}
    for i, date_step in enumerate(dates):

        # compute solar azimuth and elevation
        az, el = calculateSunAngles(hour=date_step.hour, 
                                        minute=date_step.minute, 
                                        sec=date_step.second,
                                        day=date_step.day,
                                        month=date_step.month,
                                        year=date_step.year,
                                        observerLatitude=50.9224226,
                                        observerLongitude=6.3639119,
                                        )

        # append result with velocity placeholders
        az_velocity = 0
        el_velocity = 0
        results[date_step] = Result(azimuth=az,
                                    elevation=el,
                                    azim_velocity=az_velocity,
                                    elev_velocity=el_velocity
                                    )

    ##################
    #   Velocities   #
    ##################
    for i, key in enumerate(results.keys()):
        # skip first and last date
        if i == 0 or i == len(results.keys())-1:
            continue

        # compute derivations
        diff_sec = (dates[i+1] - dates[i-1]).total_seconds()
        az_velocity = (results[dates[i+1]].azimuth - results[dates[i-1]].azimuth) / diff_sec
        el_velocity = (results[dates[i+1]].elevation - results[dates[i-1]].elevation) / diff_sec
        
        results[key] = Result(azimuth=results[key].azimuth,
                                elevation=results[key].elevation,
                                azim_velocity=az_velocity,
                                elev_velocity=el_velocity
                                )

    # remove first and last day
    results.pop(start_date)
    results.pop(dates[-1])

    ################
    #   Plotting   #
    ################

    fig = plt.figure()
    ax_1 = fig.add_subplot(3,2,1)
    ax_2 = fig.add_subplot(3,2,2)
    ax_3 = fig.add_subplot(3,2,3)
    ax_4 = fig.add_subplot(3,2,4)
    ax_5 = fig.add_subplot(3,2,5)

    summer_start = datetime.datetime(year=2022, month=6, day=21)
    summer_end = datetime.datetime(year=2022, month=6, day=22)
    winter_start = datetime.datetime(year=2022, month=12, day=21)
    winter_end = datetime.datetime(year=2022, month=12, day=22)

    ax_1.scatter([key.hour + key.minute / 60 for key in results.keys() if (key.hour > min_hour and key.hour < max_hour)], [val.azimuth for key, val in results.items() if (key.hour > min_hour and key.hour < max_hour)], c='grey')
    ax_1.scatter([key.hour + key.minute / 60 for key in results.keys() if (key >= winter_start and key <= winter_end and key.hour > min_hour and key.hour < max_hour)], [val.azimuth for key, val in results.items() if (key >= winter_start and key <= winter_end and key.hour > min_hour and key.hour < max_hour)], c='blue')
    ax_1.scatter([key.hour + key.minute / 60 for key in results.keys() if (key >= summer_start and key <= summer_end and key.hour > min_hour and key.hour < max_hour)], [val.azimuth for key, val in results.items() if (key >= summer_start and key <= summer_end and key.hour > min_hour and key.hour < max_hour)], c='orange')

    ax_2.scatter([key.hour + key.minute / 60 for key in results.keys() if (key.hour > min_hour and key.hour < max_hour)], [val.elevation for key, val in results.items() if (key.hour > min_hour and key.hour < max_hour)], c='grey')
    ax_2.scatter([key.hour + key.minute / 60 for key in results.keys() if (key >= winter_start and key <= winter_end and key.hour > min_hour and key.hour < max_hour)], [val.elevation for key, val in results.items() if (key >= winter_start and key <= winter_end and key.hour > min_hour and key.hour < max_hour)], c='blue')
    ax_2.scatter([key.hour + key.minute / 60 for key in results.keys() if (key >= summer_start and key <= summer_end and key.hour > min_hour and key.hour < max_hour)], [val.elevation for key, val in results.items() if (key >= summer_start and key <= summer_end and key.hour > min_hour and key.hour < max_hour)], c='orange')

    ax_3.scatter([key.hour + key.minute / 60 for key in results.keys() if (key.hour > min_hour and key.hour < max_hour)], [val.azim_velocity / 180 * math.pi * 1000 for key, val in results.items() if (key.hour > min_hour and key.hour < max_hour)], c='grey')
    ax_3.scatter([key.hour + key.minute / 60 for key in results.keys() if (key >= winter_start and key <= winter_end and key.hour > min_hour and key.hour < max_hour)], [val.azim_velocity / 180 * math.pi * 1000 for key, val in results.items() if (key >= winter_start and key <= winter_end and key.hour > min_hour and key.hour < max_hour)], c='blue')
    ax_3.scatter([key.hour + key.minute / 60 for key in results.keys() if (key >= summer_start and key <= summer_end and key.hour > min_hour and key.hour < max_hour)], [val.azim_velocity / 180 * math.pi * 1000 for key, val in results.items() if (key >= summer_start and key <= summer_end and key.hour > min_hour and key.hour < max_hour)], c='orange')

    ax_4.scatter([key.hour + key.minute / 60 for key in results.keys() if (key.hour > min_hour and key.hour < max_hour)], [val.elev_velocity / 180 * math.pi * 1000 for key, val in results.items() if (key.hour > min_hour and key.hour < max_hour)], c='grey')
    ax_4.scatter([key.hour + key.minute / 60 for key in results.keys() if (key >= winter_start and key <= winter_end and key.hour > min_hour and key.hour < max_hour)], [val.elev_velocity / 180 * math.pi * 1000 for key, val in results.items() if (key >= winter_start and key <= winter_end and key.hour > min_hour and key.hour < max_hour)], c='blue')
    ax_4.scatter([key.hour + key.minute / 60 for key in results.keys() if (key >= summer_start and key <= summer_end and key.hour > min_hour and key.hour < max_hour)], [val.elev_velocity / 180 * math.pi * 1000 for key, val in results.items() if (key >= summer_start and key <= summer_end and key.hour > min_hour and key.hour < max_hour)], c='orange')

    ax_5.scatter([key.hour + key.minute / 60 for key in results.keys() if (key.hour > min_hour and key.hour < max_hour)], [val.velocity / 180 * math.pi * 1000 for key, val in results.items() if (key.hour > min_hour and key.hour < max_hour)], c='grey')
    ax_5.scatter([key.hour + key.minute / 60 for key in results.keys() if (key >= winter_start and key <= winter_end and key.hour > min_hour and key.hour > min_hour and key.hour < max_hour)], [val.velocity / 180 * math.pi * 1000 for key, val in results.items() if (key >= winter_start and key <= winter_end and key.hour > min_hour and key.hour < max_hour)], c='blue')
    ax_5.scatter([key.hour + key.minute / 60 for key in results.keys() if (key >= summer_start and key <= summer_end and key.hour > min_hour and key.hour > min_hour and key.hour < max_hour)], [val.velocity / 180 * math.pi * 1000 for key, val in results.items() if (key >= summer_start and key <= summer_end and key.hour > min_hour and key.hour < max_hour)], c='orange')

    ax_1.set_xlabel('hour')
    ax_1.set_ylabel('azimuth [°]')

    ax_2.set_xlabel('hour')
    ax_2.set_ylabel('elevation [°]')

    ax_3.set_xlabel('hour')
    ax_3.set_ylabel('azimuth velocity [mRad/s]')

    ax_4.set_xlabel('hour')
    ax_4.set_ylabel('elevation velocity [mRad/s]')

    ax_5.set_xlabel('hour')
    ax_5.set_ylabel('combined velocity [mRad/s]')

    plt.show()

if __name__ == '__main__':
    main()