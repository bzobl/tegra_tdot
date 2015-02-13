#!/bin/sh
#
#                    brightness (int)    : min=0 max=255 step=1 default=128 value=128
#                      contrast (int)    : min=0 max=255 step=1 default=128 value=128
#                    saturation (int)    : min=0 max=255 step=1 default=128 value=128
#white_balance_temperature_auto (bool)   : default=1 value=1
#                          gain (int)    : min=0 max=255 step=1 default=0 value=120
#          power_line_frequency (menu)   : min=0 max=2 default=2 value=2
#		                                       0: Disabled
#		                                       1: 50 Hz
#		                                       2: 60 Hz
#     white_balance_temperature (int)    : min=2000 max=6500 step=1 default=4000 value=2866 flags=inactive
#                     sharpness (int)    : min=0 max=255 step=1 default=128 value=128
#        backlight_compensation (int)    : min=0 max=1 step=1 default=0 value=0
#                 exposure_auto (menu)   : min=0 max=3 default=3 value=3
#		                                       1: Manual Mode
#			                                     3: Aperture Priority Mode
#             exposure_absolute (int)    : min=3 max=2047 step=1 default=250 value=333 flags=inactive
#        exposure_auto_priority (bool)   : default=0 value=1
#                  pan_absolute (int)    : min=-36000 max=36000 step=3600 default=0 value=0
#                 tilt_absolute (int)    : min=-36000 max=36000 step=3600 default=0 value=0
#                focus_absolute (int)    : min=0 max=250 step=5 default=0 value=0 flags=inactive
#                    focus_auto (bool)   : default=1 value=1
#                 zoom_absolute (int)    : min=100 max=500 step=1 default=100 value=100

DEVICE=/dev/video0

v4l2-ctl -d $DEVICE -c white_balance_temperature_auto=0
v4l2-ctl -d $DEVICE -c power_line_frequency=1
v4l2-ctl -d $DEVICE -c exposure_auto=1
v4l2-ctl -d $DEVICE -c focus_auto=0
