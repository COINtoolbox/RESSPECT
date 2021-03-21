"""
    utils for fit_lighhtcurves methods
"""

SNPCC_LC_MAPPINGS = {
    "snii": ['2', '3', '4', '12', '15', '17', '19', '20', '21', '24', '25',
             '26', '27', '30', '31', '32', '33', '34', '35', '36', '37', '38',
             '39', '40', '41', '42', '43', '44'],
    "snibc": ['1', '5', '6', '7', '8', '9', '10', '11', '13', '14', '16',
              '18', '22', '23', '29', '45', '28']
}


def read_file(file_path):
    with open(file_path, "r") as file:
        lines = [line.split() for line in file.readlines()]
        return [line for line in lines if len(line) > 1]


def get_sntype(value):
    if value in SNPCC_LC_MAPPINGS["snibc"]:
        return 'Ibc'
    elif value in SNPCC_LC_MAPPINGS["snii"]:
        return 'II'
    elif value == '0':
        return 'Ia'
    raise ValueError('Unknown supernova type!')
