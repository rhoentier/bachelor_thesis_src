import can


###
#
# Taken from:
# Benjamin Bohleber. Entwicklung eines graphischen Analyse- und Auswerteframeworks für Sensordaten
# beim automatisierten Fahren, Hochschule Karlsruhe, 2021.
#
###

def classify_image() -> str:
    return_msg = get_sign()

    return return_msg


def read():
    counter = 4
    can_filter = [{"can_id": 0x833, "can_mask": 0x21}]

    can0 = can.interface.Bus(
        bustype='kvaser', channel='0', can_filters=can_filter)
    reader = can.BufferedReader()

    can.Notifier(can0, [reader])

    msg = None

    while msg is None and counter != 0:
        msg = reader.get_message()
        counter -= 1

    return msg


def get_sign():
    msg = read()
    return msg
