import numpy as np


def get_type(infoline):
    type_vec = np.zeros((12,), dtype=np.float64)
    if infoline['type'] == 'temperature':
        type_vec[0] = 1.
    elif infoline['type'] == 'windChill':
        type_vec[1] = 1.
    elif infoline['type'] == 'windSpeed':
        type_vec[2] = 1.
    elif infoline['type'] == 'windDir':
        type_vec[3] = 1.
    elif infoline['type'] == 'gust':
        type_vec[4] = 1.
    elif infoline['type'] == 'skyCover':
        type_vec[5] = 1.
    elif infoline['type'] == 'precipPotential':
        type_vec[6] = 1.
    elif infoline['type'] == 'thunderChance':
        type_vec[7] = 1.
    elif infoline['type'] == 'rainChance':
        type_vec[8] = 1.
    elif infoline['type'] == 'snowChance':
        type_vec[9] = 1.
    elif infoline['type'] == 'freezingRainChance':
        type_vec[10] = 1.
    elif infoline['type'] == 'sleetChance':
        type_vec[11] = 1.
    else:
        raise Exception('invalid type: ', infoline['type'])
    return type_vec


def get_label(infoline):
    label_vec = np.zeros((5,), dtype=np.float64)
    if infoline['label'] == 'Tonight':
        label_vec[0] = 1.
    elif infoline['label'] == 'Sunday':
        label_vec[1] = 1.
    elif infoline['label'] == 'Monday':
        label_vec[2] = 1.
    elif infoline['label'] == 'Tuesday':
        label_vec[3] = 1.
    elif infoline['label'] == 'Wednesday':
        label_vec[4] = 1.
    else:
        raise Exception('invalid label: ', infoline['label'])
    return label_vec


def get_time(infoline):
    time_vec = np.zeros((10,), dtype=np.float64)
    if infoline['time'] == '6-9':
        time_vec[0] = 1.
    elif infoline['time'] == '6-13':
        time_vec[1] = 1.
    elif infoline['time'] == '6-21':
        time_vec[2] = 1.
    elif infoline['time'] == '9-21':
        time_vec[3] = 1.
    elif infoline['time'] == '13-21':
        time_vec[4] = 1.
    elif infoline['time'] == '17-21':
        time_vec[5] = 1.
    elif infoline['time'] == '17-26':
        time_vec[6] = 1.
    elif infoline['time'] == '17-30':
        time_vec[7] = 1.
    elif infoline['time'] == '21-30':
        time_vec[8] = 1.
    elif infoline['time'] == '26-30':
        time_vec[9] = 1.
    else:
        raise Exception('invalid time: ', infoline['time'])
    return time_vec


def get_num(num_str):
    """
    get binary-array representation, for the next get_nums function.
    even without the last position for (num_str==''), min~max range is -100 to 100, 2^8 is enough.
    """
    if num_str == '':
        num_vec = np.zeros((9,), dtype=np.float64)
        num_vec[-1] = 1.
    else:
        deci = int(num_str)
        num_vec = np.zeros((9,), dtype=np.float64)
        if deci > 0:
            num_bin = bin(deci)
            bin_part = int(num_bin[2:])
            ind = 2
            while bin_part:
                num_vec[-ind] = np.float64((bin_part % 10))
                bin_part = int(bin_part / 10)
                ind += 1
        elif deci < 0:
            num_vec[0] = 1.
            num_bin = bin(deci)
            bin_part = int(num_bin[3:])
            ind = 2
            while bin_part:
                num_vec[-ind] = np.float64((bin_part % 10))
                bin_part = int(bin_part / 10)
                ind += 1
        elif deci == 0:
            pass  # num_vec will be retained all zeros when deci == 0
        else:
            pass
    return num_vec


def get_nums(min_str, mean_str, max_str):
    min_vec = get_num(min_str)
    mean_vec = get_num(mean_str)
    max_vec = get_num(max_str)
    nums_vec = np.concatenate((min_vec, mean_vec, max_vec), axis=0)
    return nums_vec


def get_temperature(infoline):
    if infoline['type'] == 'temperature':
        temp_vec = get_nums(infoline['min'], infoline['mean'], infoline['max'])
    else:
        temp_vec = get_nums('', '', '')
    return temp_vec


def get_windchill(infoline):
    if infoline['type'] == 'windChill':
        chill_vec = get_nums(infoline['min'], infoline['mean'], infoline['max'])
    else:
        chill_vec = get_nums('', '', '')
    return chill_vec


def get_windSpeed(infoline):
    if infoline['type'] == 'windSpeed':
        windSpeed_vec = get_nums(infoline['min'], infoline['mean'], infoline['max'])
    else:
        windSpeed_vec = get_nums('', '', '')
    return windSpeed_vec


def get_bucket20(infoline):
    bucket20vec = np.zeros((3,), dtype=np.float64)
    if infoline['mode_bucket_0_20_2'] == '0-10':
        bucket20vec[0] = 1.
    elif infoline['mode_bucket_0_20_2'] == '10-20':
        bucket20vec[1] = 1.
    elif infoline['mode_bucket_0_20_2'] == '':
        bucket20vec[2] = 1.
    else:
        raise Exception('invalid bucket20Vec: ', infoline['mode_bucket_0_20_2'])
    return bucket20vec


def get_dirMode(infoline):
    dir_vec = np.zeros((18,), dtype=np.float64)
    if infoline['type'] == 'windDir':
        if infoline['mode'] == '':
            dir_vec[0] = 1.
        elif infoline['mode'] == 'S':
            dir_vec[1] = 1.
        elif infoline['mode'] == 'SW':
            dir_vec[2] = 1.
        elif infoline['mode'] == 'SSE':
            dir_vec[3] = 1.
        elif infoline['mode'] == 'WSW':
            dir_vec[4] = 1.
        elif infoline['mode'] == 'ESE':
            dir_vec[5] = 1.
        elif infoline['mode'] == 'E':
            dir_vec[6] = 1.
        elif infoline['mode'] == 'W':
            dir_vec[7] = 1.
        elif infoline['mode'] == 'SE':
            dir_vec[8] = 1.
        elif infoline['mode'] == 'NE':
            dir_vec[9] = 1.
        elif infoline['mode'] == 'SSW':
            dir_vec[10] = 1.
        elif infoline['mode'] == 'NNE':
            dir_vec[11] = 1.
        elif infoline['mode'] == 'WNW':
            dir_vec[12] = 1.
        elif infoline['mode'] == 'N':
            dir_vec[13] = 1.
        elif infoline['mode'] == 'NNW':
            dir_vec[14] = 1.
        elif infoline['mode'] == 'ENE':
            dir_vec[15] = 1.
        elif infoline['mode'] == 'NW':
            dir_vec[16] = 1.
        else:
            raise Exception('invalid mode_windDir: ', infoline['mode'])
    else:
        dir_vec[-1] = 1.
    return dir_vec


def get_gust(infoline):
    if infoline['type'] == 'gust':
        gust_vec = get_nums(infoline['min'], infoline['mean'], infoline['max'])
    else:
        gust_vec = get_nums('', '', '')
    return gust_vec


def get_cover(infoline):
    cover_vec = np.zeros((5,), dtype=np.float64)
    if infoline['type'] == 'skyCover':
        if infoline['mode_bucket_0_100_4'] == '0-25':
            cover_vec[0] = 1.
        elif infoline['mode_bucket_0_100_4'] == '25-50':
            cover_vec[1] = 1.
        elif infoline['mode_bucket_0_100_4'] == '50-75':
            cover_vec[2] = 1.
        elif infoline['mode_bucket_0_100_4'] == '75-100':
            cover_vec[3] = 1.
        else:
            raise Exception('invalid mode_skyCover: ', infoline['mode_bucket_0_100_4'])
    else:
        cover_vec[-1] = 1.
    return cover_vec


def get_prec(infoline):
    if infoline['type'] == 'precipPotential':
        prec_vec = get_nums(infoline['min'], infoline['mean'], infoline['max'])
    else:
        prec_vec = get_nums('', '', '')
    return prec_vec


def get_thunderMode(infoline):
    thunder_vec = np.zeros((6,), dtype=np.float64)
    if infoline['type'] == 'thunderChance':
        if infoline['mode'] == '--':
            thunder_vec[0] = 1.
        elif infoline['mode'] == 'SChc':
            thunder_vec[1] = 1.
        elif infoline['mode'] == 'Chc':
            thunder_vec[2] = 1.
        elif infoline['mode'] == 'Lkly':
            thunder_vec[3] = 1.
        elif infoline['mode'] == 'Def':
            thunder_vec[4] = 1.
        else:
            raise Exception('invalid mode_thunderChance: ', infoline['mode'])
    else:
        thunder_vec[-1] = 1.
    return thunder_vec


def get_rainMode(infoline):
    rain_vec = np.zeros((6,), dtype=np.float64)
    if infoline['type'] == 'rainChance':
        if infoline['mode'] == '--':
            rain_vec[0] = 1.
        elif infoline['mode'] == 'SChc':
            rain_vec[1] = 1.
        elif infoline['mode'] == 'Chc':
            rain_vec[2] = 1.
        elif infoline['mode'] == 'Lkly':
            rain_vec[3] = 1.
        elif infoline['mode'] == 'Def':
            rain_vec[4] = 1.
        else:
            raise Exception('invalid mode_rainChance: ', infoline['mode'])
    else:
        rain_vec[-1] = 1.
    return rain_vec


def get_snowMode(infoline):
    snow_vec = np.zeros((6,), dtype=np.float64)
    if infoline['type'] == 'snowChance':
        if infoline['mode'] == '--':
            snow_vec[0] = 1.
        elif infoline['mode'] == 'SChc':
            snow_vec[1] = 1.
        elif infoline['mode'] == 'Chc':
            snow_vec[2] = 1.
        elif infoline['mode'] == 'Lkly':
            snow_vec[3] = 1.
        elif infoline['mode'] == 'Def':
            snow_vec[4] = 1.
        else:
            raise Exception('invalid mode_snowChance: ', infoline['mode'])
    else:
        snow_vec[-1] = 1.
    return snow_vec


def get_freezeMode(infoline):
    freeze_vec = np.zeros((6,), dtype=np.float64)
    if infoline['type'] == 'freezingRainChance':
        if infoline['mode'] == '--':
            freeze_vec[0] = 1.
        elif infoline['mode'] == 'SChc':
            freeze_vec[1] = 1.
        elif infoline['mode'] == 'Chc':
            freeze_vec[2] = 1.
        elif infoline['mode'] == 'Lkly':
            freeze_vec[3] = 1.
        elif infoline['mode'] == 'Def':
            freeze_vec[4] = 1.
        else:
            raise Exception('invalid mode_freezingRainChance: ', infoline['mode'])
    else:
        freeze_vec[-1] = 1.
    return freeze_vec


def get_sleetMode(infoline):
    sleet_vec = np.zeros((6,), dtype=np.float64)
    if infoline['type'] == 'sleetChance':
        if infoline['mode'] == '--':
            sleet_vec[0] = 1.
        elif infoline['mode'] == 'SChc':
            sleet_vec[1] = 1.
        elif infoline['mode'] == 'Chc':
            sleet_vec[2] = 1.
        elif infoline['mode'] == 'Lkly':
            sleet_vec[3] = 1.
        elif infoline['mode'] == 'Def':
            sleet_vec[4] = 1.
        else:
            raise Exception('invalid mode_sleetChance: ', infoline['mode'])
    else:
        sleet_vec[-1] = 1.
    return sleet_vec


def get_infovec(infoline):
    type_vec = get_type(infoline)
    label_vec = get_label(infoline)
    time_vec = get_time(infoline)
    temp_vec = get_temperature(infoline)
    chill_vec = get_windchill(infoline)
    speed_vec = get_windSpeed(infoline)
    bucket20vec = get_bucket20(infoline)
    dir_vec = get_dirMode(infoline)
    gust_vec = get_gust(infoline)
    cover_vec = get_cover(infoline)
    prec_vec = get_prec(infoline)
    thunder_vec = get_thunderMode(infoline)
    rain_vec = get_rainMode(infoline)
    snow_vec = get_snowMode(infoline)
    freeze_vec = get_freezeMode(infoline)
    sleet_vec = get_sleetMode(infoline)
    info_vec = np.concatenate((type_vec, label_vec, time_vec, temp_vec, chill_vec, speed_vec,
                               bucket20vec, dir_vec, gust_vec, cover_vec, prec_vec, thunder_vec,
                               rain_vec, snow_vec, freeze_vec, sleet_vec), axis=0)
    return info_vec


def get_info(batch_data):
    info_matrix = []
    for the_id in range(36):
        infoline = batch_data['id' + str(the_id)]
        info_matrix.append(get_infovec(infoline))
    return np.float64(np.array(info_matrix))  # convert list into numpy_array
