import matplotlib.pyplot as plt
import matplotlib
import datetime
import math
from typing import Tuple
import os

import pvlib
from pvlib.location import Location
import pandas as pd
from zoneinfo import ZoneInfo
import matplotlib as mpl

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

def addAngles(angle_plot, date_plot, start_date, end_date, minute_delta, type, lat, lng, min_start_date, max_end_date):
    color_dict = {
        'train' : 'orange',
        'validation' : 'green',
        'test' : [51.0/255.0, 153.0/255, 1.0],
    }

    marker_dict = {
        'train' : '*',
        'validation' : 'D',
        'test' : '.',
    }

    elev_angles = []
    azim_angles = []
    dates = []
    hours = []
    date_deltas = []
    date : datetime.datetime = start_date
    while date <= end_date:

        # compute solar azimuth and elevation
        # az, el = calculateSunAngles(hour=date.hour, 
        #                                 minute=date.minute, 
        #                                 sec=date.second,
        #                                 day=date.day,
        #                                 month=date.month,
        #                                 year=date.year,
        #                                 observerLatitude=lat,
        #                                 observerLongitude=lng,
        #                                 )
        # elev_angles.append(90-el)
        # azim_angles.append(180 - az)
        
        site = Location(latitude=lat, longitude=lng)
        # Estimate Solar Position with the 'Location' object
        times = pd.date_range(start=date, end=date)
        solpos = site.get_solarposition(times)
        az = solpos.azimuth[0]
        el = solpos.elevation[0]

        # Visualize the resulting DataFrame
        # print(solpos.head())

        elev_angles.append(el)
        azim_angles.append(az)

        dates.append(datetime.date(year=date.year, month=date.month, day=date.day))
        date_deltas.append((date - min_start_date) / (max_end_date - min_start_date))
        hours.append(date.hour + date.minute / 60.0)

        date = date + datetime.timedelta(minutes=minute_delta)

    cmap = plt.get_cmap('winter')

    angle_plot.scatter(azim_angles, elev_angles, color=color_dict[type], marker=marker_dict[type])
    date_plot.scatter(dates, hours, color=color_dict[type], marker=marker_dict[type])

def main(start_dates, end_dates, minute_deltas, types, lat=50.92242, lng=6.36391, tmz='GMT+1'):

    fig = plt.figure(figsize=(35.4,12))
    ax = []
    for i in range(6):
        ax.append(fig.add_subplot(2,3,i+1))

    for i in range(3):
        # sd = start_dates[i]
        # ed = end_dates[i]
        # md = minute_deltas[i]
        # t = types[i]
        lt = lat[i]
        lg = lng[i]
        t = tmz[i]
        min_start_date = min(start_dates[i])
        max_end_date = max(end_dates[i])
        for sd, ed, md, type in zip(start_dates[i], end_dates[i], minute_deltas[i], types[i]):
            addAngles(angle_plot=ax[i+3], date_plot=ax[i], start_date=sd, end_date=ed, minute_delta=md, type=type, lat=lt, lng=lg,  min_start_date=min_start_date, max_end_date=max_end_date)
        ax[i+3].set_xlim((75,275))
        ax[i+3].set_ylim((0,80))
        ax[i].set_ylim((7.5,18))

        ax[i].set_title('Lat: ' + str(lt) + ' Lng: ' + str(lg) + ' TMZ: ' + t)

        if i > 0:
            ax[i].set_yticks([])
            ax[i+3].set_yticks([])

            plt.setp(ax[i+3].get_xticklabels()[0], visible=False)
            # plt.setp(ax[i+3].get_xticklabels()[-1], visible=False)

    ax[0].xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m-%d'))
    ax[0].xaxis.set_major_locator(mpl.dates.DayLocator(interval=10))
    # ax[0].text(0, 0, '(a)')
    ax[0].set_ylabel('Hour')

    ax[1].xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m-%d'))
    ax[1].xaxis.set_major_locator(mpl.dates.DayLocator(interval=60))
    # ax[1].text(0, 0, '(b)')
    ax[1].set_xlabel('Date')

    plt.text(0.01, 0.99 , '(a)',
     horizontalalignment='left',
     verticalalignment='top',
     transform = ax[0].transAxes
     )
    
    plt.text(0.01, 0.99 , '(b)',
     horizontalalignment='left',
     verticalalignment='top',
     transform = ax[1].transAxes
     )
    
    plt.text(0.01, 0.99 , '(c)',
     horizontalalignment='left',
     verticalalignment='top',
     transform = ax[2].transAxes
     )

    ax[2].xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m-%d'))
    ax[2].xaxis.set_major_locator(mpl.dates.DayLocator(interval=10))
    # ax[2].text(0, 0, '(c)')

    ax[3].set_ylabel('Solar Elevation [Deg]')
    ax[4].set_xlabel('Solar Azimuth [Deg]')

    ax[5].scatter(-1, -1, color='orange', marker='*', s=150, label='Training')
    ax[5].scatter(-1, -1, color=[51.0/255.0, 153.0/255, 1.0], marker='.', s=150, label='Testing')
    ax[5].legend()


    fig.tight_layout()
    fig.subplots_adjust(wspace=0)

    # for sd, ed, md, type in zip(start_dates, end_dates, minute_deltas, types):
    #     addAngles(angle_plot=ax_1, date_plot=ax_2, start_date=sd, end_date=ed, minute_delta=md, type=type, lat=lat, lng=lng)

    # ax_1.scatter([], [], color='blue', label='train/validation')
    # ax_1.scatter([], [], color='orange', label='test')
    # ax_1.set_xlim((70, 265))
    # ax_1.set_xlim((-180, 180))
    # ax_1.set_ylim((0, 90))
    # ax_1.set_ylim((-180, 180))
    # ax_1.set_xlabel('Solar Azimuth [°]')
    # ax_1.set_ylabel('Solar Elevation [°]')
    # ax_1.legend()

    # ax_2.scatter([], [], color='blue', label='train/validation')
    # ax_2.scatter([], [], color='orange', label='test')
    # ax_2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d'))
    # ax_2.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=60))
    # ax_2.xaxis.set_ticklabels([start_dates[0], start_dates[-1]])
    # ax_2.set_ylim((7, 20))
    # ax_2.set_xlabel('Date')
    # ax_2.set_ylabel('Hour')
    # ax_2.legend()

    # ax_2.set_title('Lat: ' + str(lat) + ' Lng: ' + str(lng) + ' TMZ: ' + tmz)

    plt.savefig(os.path.abspath(os.path.join('/Users/moritz/Desktop', str(datetime.datetime.now())+ '.pdf')), bbox_inches='tight')

if __name__ == '__main__':
    matplotlib.rcParams.update({'font.size': 24})

    smith_zone = 'US/Mountain'
    sarr_zone = 'Etc/GMT-0'

    start_dates_pargmann = [
        # pargmann
        datetime.datetime(year=2020, month=8, day= 1, hour=9),
        datetime.datetime(year=2020, month=8, day= 12, hour=9),
        datetime.datetime(year=2020, month=8, day= 13, hour=9),
        datetime.datetime(year=2020, month=8, day= 14, hour=9),
        datetime.datetime(year=2020, month=8, day= 26, hour=9),
        datetime.datetime(year=2020, month=8, day= 27, hour=9),
        datetime.datetime(year=2020, month=8, day= 30, hour=9),
        datetime.datetime(year=2020, month=9, day= 2, hour=9),
    ]

    start_dates_sarr = [
        # Sarr
        datetime.datetime(year=2011, month=6, day= 16, hour=12, minute=30, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2011, month=6, day= 17, hour=9, minute=30, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2011, month=7, day= 15, hour=12, tzinfo=ZoneInfo(smith_zone)),
    ]

    start_dates_smith = [
        # Smith
        datetime.datetime(year=2012, month=8, day= 9, hour=8, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2012, month=8, day= 11, hour=8, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2012, month=8, day= 13, hour=8, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2012, month=8, day= 15, hour=8, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2012, month=8, day= 17, hour=8, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2012, month=8, day= 19, hour=8, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2012, month=8, day= 21, hour=8, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2012, month=8, day= 23, hour=8, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2012, month=8, day= 25, hour=8, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2012, month=8, day= 27, hour=8, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2012, month=10, day= 22, hour=8, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2012, month=10, day= 24, hour=8, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2012, month=10, day= 26, hour=8, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2012, month=10, day= 28, hour=8, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2013, month=2, day= 4, hour=8, minute=00, tzinfo=ZoneInfo(smith_zone)),

        datetime.datetime(year=2012, month=10, day= 30, hour=8, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2013, month=2, day= 12, hour=8, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2013, month=2, day= 14, hour=8, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2013, month=2, day= 16, hour=8, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2013, month=2, day= 18, hour=8, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2013, month=2, day= 20, hour=8, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2013, month=2, day= 22, hour=8, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2013, month=2, day= 24, hour=8, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2013, month=2, day= 26, hour=8, minute=00, tzinfo=ZoneInfo(smith_zone)),
    ]

    end_dates_pargmann = [
        # pargmann
        datetime.datetime(year=2020, month=8, day= 1, hour=15, minute=30),
        datetime.datetime(year=2020, month=8, day= 12, hour=15, minute=30),
        datetime.datetime(year=2020, month=8, day= 13, hour=15, minute=30),
        datetime.datetime(year=2020, month=8, day= 14, hour=15, minute=30),
        datetime.datetime(year=2020, month=8, day= 26, hour=15, minute=30),
        datetime.datetime(year=2020, month=8, day= 27, hour=15, minute=30),
        datetime.datetime(year=2020, month=8, day= 30, hour=15, minute=30),
        datetime.datetime(year=2020, month=8, day= 2, hour=15, minute=30),
    ]

    end_dates_sarr = [
        # Sarr
        datetime.datetime(year=2011, month=6, day= 16, hour=16, minute=40, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2011, month=6, day= 17, hour=15, minute=55, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2011, month=7, day= 15, hour=16, minute=30, tzinfo=ZoneInfo(smith_zone)),
    ]

    end_dates_smith = [
        # Smith
        datetime.datetime(year=2012, month=8, day= 9, hour=16, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2012, month=8, day= 11, hour=16, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2012, month=8, day= 13, hour=16, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2012, month=8, day= 15, hour=16, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2012, month=8, day= 17, hour=16, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2012, month=8, day= 19, hour=16, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2012, month=8, day= 21, hour=16, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2012, month=8, day= 23, hour=16, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2012, month=8, day= 25, hour=16, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2012, month=8, day= 27, hour=16, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2012, month=10, day= 22, hour=16, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2012, month=10, day= 24, hour=16, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2012, month=10, day= 26, hour=16, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2012, month=10, day= 28, hour=16, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2013, month=2, day= 4, hour=16, minute=00, tzinfo=ZoneInfo(smith_zone)),

        datetime.datetime(year=2012, month=10, day= 30, hour=16, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2013, month=2, day= 12, hour=16, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2013, month=2, day= 14, hour=16, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2013, month=2, day= 16, hour=16, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2013, month=2, day= 18, hour=16, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2013, month=2, day= 20, hour=16, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2013, month=2, day= 22, hour=16, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2013, month=2, day= 24, hour=16, minute=00, tzinfo=ZoneInfo(smith_zone)),
        datetime.datetime(year=2013, month=2, day= 26, hour=16, minute=00, tzinfo=ZoneInfo(smith_zone)),
    ]

    minute_deltas_pargmann = [
        # pargmann
        3,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
    ]

    minute_deltas_sarr = [
        # Sarr
        20,
        20,
        4,
    ]

    minute_deltas_smith = [
        # Smith
        60,
        60,
        60,
        60,
        60,
        60,
        60,
        60,
        60,
        60,
        60,
        60,
        60,
        60,
        60,

        60,
        60,
        60,
        60,
        60,
        60,
        60,
        60,
        60,
    ]

    types_parmann = [
        # pargmann
        'train',
        'train',
        'train',
        'train',
        'test',
        'test',
        'test',
        'test',
    ]

    types_sarr = [
        # Sarr
        'train',
        'train',
        'test',
    ]

    types_smith = [
        # Smith
        'train',
        'train',
        'train',
        'train',
        'train',
        'train',
        'train',
        'train',
        'train',
        'train',
        'train',
        'train',
        'train',
        'train',
        'train',

        'test',
        'test',
        'test',
        'test',
        'test',
        'test',
        'test',
        'test',
        'test',
    ]

    # Sarr
    # lat=14.69579 
    # lng=360-16.47936
    # tmz='UTC+0'

    # Smith
    coord_a = [35.05390, 360-106.52849, 'UTC-7']
    # tmz='UTC+1'

    main(start_dates=[start_dates_sarr, start_dates_smith, start_dates_pargmann], 
         end_dates=[end_dates_sarr, end_dates_smith, end_dates_pargmann], 
         minute_deltas=[minute_deltas_sarr, minute_deltas_smith, minute_deltas_pargmann], 
         types=[types_sarr, types_smith, types_parmann], 
         lat=[coord_a[0], coord_a[0], 50.92242], 
         lng=[coord_a[1], coord_a[1], 6.36391], 
         tmz=[coord_a[2], coord_a[2], 'UTC+1'])